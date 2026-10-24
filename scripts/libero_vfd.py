"""Velocity disagreement and calibration helpers for GR00T LIBERO monitoring.

This module deliberately does not change the existing inference/evaluation path.
Two epochs of one run are supported as a snapshot experiment, not an independent
deep ensemble. Scores use the acting model's flow trajectory (VFD paper, Eq. 42).
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def flow_grid(num_steps, flow_times=None):
    if num_steps < 1:
        raise ValueError("denoising steps must be positive")
    times = np.arange(num_steps, dtype=float) / num_steps
    selected = list(range(num_steps))
    if flow_times is not None:
        selected = []
        for value in flow_times:
            if not 0 <= value < 1:
                raise ValueError("flow times must be in [0, 1)")
            index = int(round(value * num_steps))
            if index >= num_steps or not np.isclose(times[index], value):
                raise ValueError(f"flow time {value} is not on the {num_steps}-step inference grid")
            selected.append(index)
        selected = sorted(set(selected))
        if not selected:
            raise ValueError("select at least one flow time")
    chosen = times[selected]
    weights = chosen / (1 - chosen) * np.diff(np.r_[chosen, 1.0])
    if not np.any(weights > 0):
        raise ValueError("at least one nonzero flow time is required for VFD")
    return dict(zip(selected, weights.tolist()))


def integrate_vfd(actor_velocity, scorer_velocity, noise, num_steps, action_dim=7, flow_times=None):
    """Integrate only the actor; compare both fields at exactly the same x and t.

    Padding is excluded from the score, but full-dimensional trajectories are
    integrated, matching the policy. Accumulation is float32 under BF16 inference.
    Returns terminal actions and one disagreement score per noise sample.
    """
    import torch

    if noise.ndim != 3 or not 0 < action_dim <= noise.shape[-1]:
        raise ValueError("expected noise [samples, horizon, padded_dim] and valid action_dim")
    weights = flow_grid(num_steps, flow_times)
    x = noise
    scores = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    for index in range(num_steps):
        t = index / num_steps
        velocity = actor_velocity(x, t)
        if index in weights and weights[index] > 0:
            other = scorer_velocity(x, t)
            difference = velocity[..., :action_dim].float() - other[..., :action_dim].float()
            scores += weights[index] * difference.square().sum(dim=(1, 2))
        x = x + (1.0 / num_steps) * velocity
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite VFD score")
    return x, scores


class SnapshotVFDPolicy:
    """Two checkpoints with matching normalization and LIBERO's 7 real action dims.

    The actual actor action is generated separately with batch size one and the
    original global RNG. Extra uncertainty samples use their own RNG, so changing
    sample count does not consume the actor's random stream or batch its forward.
    """

    def __init__(
        self,
        actor_path,
        scorer_path,
        data_config,
        denoising_steps=8,
        num_action_samples=1,
        flow_times=None,
        uncertainty_seed=1234,
    ):
        import torch
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy

        self.actor_path = Path(actor_path).resolve(strict=True)
        self.scorer_path = Path(scorer_path).resolve(strict=True)
        if self.actor_path == self.scorer_path:
            raise ValueError("actor and scorer must be different checkpoints")
        if num_action_samples < 1:
            raise ValueError("num_action_samples must be positive")
        flow_grid(denoising_steps, flow_times)
        metadata = []
        for path in (self.actor_path, self.scorer_path):
            values = json.loads((path / "experiment_cfg/metadata.json").read_text())
            metadata.append(values["new_embodiment"])
        if metadata[0] != metadata[1]:
            raise ValueError(
                "checkpoint normalization metadata differs; velocities need a common coordinate system"
            )
        config = load_data_config(data_config)
        self.actor, self.scorer = [
            Gr00tPolicy(
                model_path=str(path),
                embodiment_tag="new_embodiment",
                modality_config=config.modality_config(),
                modality_transform=config.transform(),
                denoising_steps=denoising_steps,
            )
            for path in (self.actor_path, self.scorer_path)
        ]
        heads = [policy.model.action_head for policy in (self.actor, self.scorer)]
        for key in ("action_dim", "action_horizon", "num_timestep_buckets"):
            if getattr(heads[0], key) != getattr(heads[1], key):
                raise ValueError(f"incompatible checkpoint heads: {key}")
        self.num_steps = denoising_steps
        self.num_samples = num_action_samples
        self.flow_times = flow_times
        self.generator = torch.Generator(device=self.actor.device).manual_seed(uncertainty_seed)
        self.signature = {
            "actor": str(self.actor_path),
            "scorer": str(self.scorer_path),
            "data_config": data_config,
            "denoising_steps": denoising_steps,
            "num_action_samples": num_action_samples,
            "flow_times": [index / denoising_steps for index in flow_grid(denoising_steps, flow_times)],
            "real_action_dim": 7,
            "score": "one_way_vfd_ot_squared_l2",
            "metadata_sha256": hashlib.sha256(json.dumps(metadata[0], sort_keys=True).encode()).hexdigest(),
        }

    @staticmethod
    def velocity_function(policy, normalized_input):
        """Mirror the checkpoint's get_action conditioning and velocity prediction."""
        import torch

        model = policy.model
        head = model.action_head
        backbone_input, action_input = model.prepare_input(normalized_input)
        backbone = head.process_backbone_output(model.backbone(backbone_input))
        visual = backbone.backbone_features
        embodiment = action_input.embodiment_id
        state = head.state_encoder(action_input.state, embodiment)

        def velocity(actions, time):
            count = actions.shape[0]
            ids = embodiment.expand(count)
            times = torch.full((count,), int(time * head.num_timestep_buckets), device=actions.device)
            features = head.action_encoder(actions, times, ids)
            if head.config.add_pos_embed:
                positions = torch.arange(features.shape[1], dtype=torch.long, device=actions.device)
                features = features + head.position_embedding(positions).unsqueeze(0)
            future = head.future_tokens.weight.unsqueeze(0).expand(count, -1, -1)
            hidden = torch.cat((state.expand(count, -1, -1), future, features), dim=1)
            output = head.model(
                hidden_states=hidden, encoder_hidden_states=visual.expand(count, -1, -1), timestep=times
            )
            return head.action_decoder(output, ids)[:, -head.action_horizon :]

        return velocity, visual

    def get_action_with_vfd(self, observation):
        import torch
        from gr00t.model.policy import COMPUTE_DTYPE, squeeze_dict_values, unsqueeze_dict_values

        if self.actor._check_state_is_batched(observation):
            raise ValueError("monitor expects a single unbatched LIBERO observation")
        inputs = unsqueeze_dict_values(observation.copy())
        inputs = {key: np.asarray(value) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            normalized = self.actor.apply_transforms(inputs)
            actor_velocity, features = self.velocity_function(self.actor, normalized)
            scorer_velocity, _ = self.velocity_function(self.scorer, normalized)
            head = self.actor.model.action_head
            noise = torch.randn(
                (1, head.action_horizon, head.action_dim), device=features.device, dtype=features.dtype
            )
            terminal, scores = integrate_vfd(
                actor_velocity, scorer_velocity, noise, self.num_steps, flow_times=self.flow_times
            )
            # Keep the acting sample's batch size and RNG identical to normal inference.
            if self.num_samples > 1:
                extra_noise = torch.randn(
                    (self.num_samples - 1, head.action_horizon, head.action_dim),
                    device=features.device,
                    dtype=features.dtype,
                    generator=self.generator,
                )
                _, extra_scores = integrate_vfd(
                    actor_velocity, scorer_velocity, extra_noise, self.num_steps, flow_times=self.flow_times
                )
                scores = torch.cat((scores, extra_scores))
            result = self.actor._get_unnormalized_action(terminal.float())
        return squeeze_dict_values(result), float(scores.mean()), scores.cpu().numpy()


def rollout_id(metadata):
    # Different simulator/noise seeds with the same official initial state are
    # conservatively treated as overlapping calibration/test configurations.
    return f"{metadata['suite']}/task{metadata['task_id']:02d}/init{metadata['initial_state_index']:03d}"


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    episodes = []
    for metadata_path in sorted(path.glob("task*/episode*.json")):
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("complete", False):
            continue
        with np.load(metadata_path.with_suffix(".npz"), allow_pickle=False) as arrays:
            scores = arrays["uncertainty"].copy()
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError(f"empty/non-finite uncertainty in {metadata_path}")
        episodes.append((metadata_path, metadata, scores))
    return manifest, episodes


def calibrate(run_paths, output, quantile=0.9, min_successes=20, scope="task"):
    """Fit the empirical quantile of successful rollout maxima, without test data.

    This follows the paper's empirical quantile convention. It does not promise
    exact finite-sample coverage or transfer guarantees across LOSO domains.
    """
    if not 0 < quantile < 1 or min_successes < 2 or scope not in ("task", "global"):
        raise ValueError("invalid calibration quantile/count/scope")
    groups, ids, signatures, suites = {}, set(), [], set()
    for path in run_paths:
        manifest, episodes = load_run(path)
        if manifest.get("smoke_only"):
            raise ValueError("truncated smoke rollouts cannot calibrate a deployment detector")
        signatures.append(manifest["score_signature"])
        for _, metadata, scores in episodes:
            identity = rollout_id(metadata)
            if identity in ids:
                raise ValueError(f"duplicate calibration initial state: {identity}")
            ids.add(identity)
            suites.add(metadata["suite"])
            key = f"{metadata['suite']}/task{metadata['task_id']:02d}" if scope == "task" else "global"
            groups.setdefault(key, [])
            if metadata["success"]:
                groups[key].append(float(np.max(scores)))
    if not signatures or not groups or any(s != signatures[0] for s in signatures):
        raise ValueError("calibration runs must use one identical actor/scorer/score configuration")
    thresholds = {}
    for key, values in groups.items():
        if len(values) < min_successes:
            raise ValueError(f"{key}: only {len(values)} successful rollouts; need {min_successes}")
        thresholds[key] = {"threshold": float(np.quantile(values, quantile)), "successful_rollouts": len(values)}
    result = {
        "score_signature": signatures[0],
        "scope": scope,
        "quantile": quantile,
        "thresholds": thresholds,
        "calibration_ids": sorted(ids),
        "calibration_suites": sorted(suites),
        "source_runs": [str(Path(p).resolve()) for p in run_paths],
        "note": "Target-suite calibration uses held-out data. Source-only global thresholds have no target coverage guarantee.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def get_threshold(calibration, signature, metadata, check_overlap=True):
    if calibration["score_signature"] != signature:
        raise ValueError("calibration actor/scorer/score configuration does not match")
    if check_overlap and rollout_id(metadata) in calibration["calibration_ids"]:
        raise ValueError(f"calibration/test initial-state overlap: {rollout_id(metadata)}")
    key = f"{metadata['suite']}/task{metadata['task_id']:02d}" if calibration["scope"] == "task" else "global"
    if key not in calibration["thresholds"]:
        raise ValueError(f"no calibrated threshold for {key}; source-only transfer requires global calibration")
    return float(calibration["thresholds"][key]["threshold"])


def detection_metrics(episodes):
    """Episodes are (success, uncertainty array, threshold); failure is positive."""
    if not episodes:
        return {"episodes": 0}
    successes = np.array([success for success, _, _ in episodes], dtype=bool)
    alarms = [np.flatnonzero(scores > threshold) for _, scores, threshold in episodes]
    flagged = np.array([len(indices) > 0 for indices in alarms])
    n_success, n_fail = int(successes.sum()), int((~successes).sum())
    tpr = float(flagged[~successes].mean()) if n_fail else None
    tnr = float((~flagged[successes]).mean()) if n_success else None
    max_length = max(len(scores) for _, scores, _ in episodes)
    detected_times = [
        float(indices[0] / max(max_length - 1, 1))
        for success, indices in zip(successes, alarms)
        if not success and len(indices)
    ]
    balanced = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    twa = (sum(1 - time for time in detected_times) / n_fail + tnr) / 2 if balanced is not None else None
    # AUROC is P(score_failure > score_success), counting ties as one half.
    # Rank statistics avoid adding scikit-learn to the GR00T environment.
    auroc = None
    if balanced is not None:
        peaks = np.asarray([np.max(scores) for _, scores, _ in episodes])
        order = np.argsort(peaks, kind="stable")
        ranks = np.empty(len(peaks), dtype=float)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and peaks[order[end]] == peaks[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auroc = float((ranks[~successes].sum() - n_fail * (n_fail + 1) / 2) / (n_fail * n_success))
    return {
        "episodes": len(episodes),
        "successes": n_success,
        "failures": n_fail,
        "TPR": tpr,
        "TNR": tnr,
        "balanced_accuracy": balanced,
        "TWA": twa,
        "AUROC": auroc,
        "mean_detection_time": float(np.mean(detected_times)) if detected_times else None,
    }
