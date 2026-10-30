"""Two-checkpoint velocity disagreement conditioned on one RA-VLA context.

This ports Part A's one-way VFD quadrature to RA-VLA's deterministic retrieved-
action source. It is a diagnostic score, not a Gaussian-source KL estimate.
Only the actor trajectory is integrated; neither retrieval nor the backbone is
repeated for the second member. Each member keeps its own trainable projectors.
"""

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT = "new_embodiment"


def flow_grid(num_steps, flow_times=None):
    """Part A / VFD Eq. 42: kappa(t) times spacing to the next selected time."""
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("denoising steps must be a positive integer")
    times = np.arange(num_steps, dtype=float) / num_steps
    selected = list(range(num_steps))
    if flow_times is not None:
        selected = []
        for value in flow_times:
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError("flow times must be finite and in [0, 1)")
            index = int(round(value * num_steps))
            if index >= num_steps or not np.isclose(times[index], value):
                raise ValueError(
                    f"flow time {value} is not on the {num_steps}-step grid"
                )
            selected.append(index)
        selected = sorted(set(selected))
    chosen = times[selected]
    weights = chosen / (1 - chosen) * np.diff(np.r_[chosen, 1.0])
    if not np.any(weights > 0):
        raise ValueError("VFD requires at least one nonzero flow time")
    return dict(zip(selected, weights.tolist()))


def integrate_context_vfd(
    actor_velocity,
    scorer_velocity,
    initial,
    num_steps,
    real_action_dim=7,
    flow_times=None,
):
    """Integrate the actor in full padded coordinates; score only real actions.

    Both velocities receive the exact same actor state and time. The scorer
    never advances its own trajectory. FP32 differences/accumulation avoid BF16
    cancellation and overflow in the diagnostic, while actor updates retain
    the original inference dtype and order of operations.
    """
    if initial.ndim != 3 or not 0 < real_action_dim <= initial.shape[-1]:
        raise ValueError(
            "expected initial [batch, horizon, padded_dim] and valid action dimension"
        )
    grid = flow_grid(num_steps, flow_times)
    selected = [i for i, weight in grid.items() if weight > 0]
    x = initial
    differences = []
    dimension_differences = []
    for index in range(num_steps):
        time = index / num_steps
        velocity = actor_velocity(x, time)
        if velocity.shape != x.shape:
            raise ValueError("actor velocity shape differs from action shape")
        if index in selected:
            other = scorer_velocity(x, time)
            if other.shape != x.shape:
                raise ValueError("scorer velocity shape differs from action shape")
            delta = (
                velocity[..., :real_action_dim].float()
                - other[..., :real_action_dim].float()
            )
            differences.append(delta.square().sum(dim=(1, 2)))
            dimension_differences.append(delta.square().sum(dim=1))
        x = x + (1.0 / num_steps) * velocity
    squared_l2 = torch.stack(differences, dim=1)
    weights = torch.tensor(
        [grid[i] for i in selected], dtype=torch.float32, device=x.device
    )
    weighted = squared_l2 * weights
    scores = weighted.sum(dim=1)
    if not torch.isfinite(scores).all() or not torch.isfinite(x).all():
        raise FloatingPointError("non-finite context VFD score or actor action")
    return (
        x,
        scores,
        {
            "flow_times": [i / num_steps for i in selected],
            "weights": weights,
            "squared_l2": squared_l2,
            "weighted_disagreement": weighted,
            "per_dimension": (
                torch.stack(dimension_differences, dim=1) * weights[None, :, None]
            ).sum(1),
        },
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _metadata(path):
    value = torch.load(
        Path(path) / "metadata.pt", map_location="cpu", weights_only=False
    )[EMBODIMENT]
    # Pickle bytes can differ even when normalization statistics are identical.
    return value, value.model_dump(mode="json")


def _weight_map(directory):
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        result = json.loads(index.read_text())["weight_map"]
    else:
        from safetensors import safe_open

        with safe_open(
            directory / "model.safetensors", framework="pt", device="cpu"
        ) as file:
            result = {key: "model.safetensors" for key in file.keys()}
    if not result or any(
        not (directory / shard).is_file() for shard in set(result.values())
    ):
        raise ValueError(f"Missing or empty safetensors checkpoint: {directory}")
    return result


def _tensor_digest(directory, prefix):
    """Hash tensor contents without materializing a second full backbone/model."""
    from contextlib import ExitStack
    from safetensors import safe_open

    directory = Path(directory)
    mapping = _weight_map(directory)
    keys = sorted(key for key in mapping if key.startswith(prefix))
    if not keys:
        raise ValueError(f"No {prefix} tensors in {directory}")
    digest = hashlib.sha256()
    with ExitStack() as stack:
        files = {
            name: stack.enter_context(
                safe_open(directory / name, framework="pt", device="cpu")
            )
            for name in sorted({mapping[key] for key in keys})
        }
        for key in keys:
            tensor = files[mapping[key]].get_tensor(key)
            digest.update(
                json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode()
            )
            # A view into the mmap, including BF16, without a giant .tobytes() copy.
            data = tensor.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(data))
    return digest.hexdigest()


def validate_checkpoint_pair(actor_path, scorer_path, retriever_path):
    """Reject incompatible normalization/backbones before sharing any features."""
    paths = [
        Path(p).resolve(strict=True) for p in (actor_path, scorer_path, retriever_path)
    ]
    actor, scorer, retriever = paths
    if actor == scorer:
        raise ValueError("actor and scorer must be different checkpoints")
    configs = [json.loads((p / "config.json").read_text()) for p in (actor, scorer)]
    for key in ("backbone_cfg", "action_head_cfg", "action_horizon", "action_dim"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"checkpoint configuration differs: {key}")
    metadata = [_metadata(p) for p in paths]
    if any(value[1] != metadata[0][1] for value in metadata[1:]):
        raise ValueError("actor/scorer/retriever normalization metadata differs")
    backbone_hashes = [_tensor_digest(p, "backbone.") for p in (actor, scorer)]
    if backbone_hashes[0] != backbone_hashes[1]:
        raise ValueError(
            "checkpoint backbones differ; sharing frozen features would change the scorer"
        )
    head_hashes = [_tensor_digest(p, "action_head.") for p in (actor, scorer)]
    if head_hashes[0] == head_hashes[1]:
        raise ValueError(
            "action heads are identical; these checkpoints cannot form an ensemble"
        )
    return {
        "actor_path": str(actor),
        "scorer_path": str(scorer),
        "retriever_path": str(retriever),
        "normalization_sha256": _digest(metadata[0][1]),
        "shared_backbone_sha256": backbone_hashes[0],
        "actor_head_sha256": head_hashes[0],
        "scorer_head_sha256": head_hashes[1],
        "model_config_sha256": _digest(configs[0]),
        "retriever_config_sha256": _digest(
            json.loads((retriever / "config.json").read_text())
        ),
    }


def load_scorer_head(actor_head, scorer_path):
    """Load only action-head tensors, retaining just one VLM in GPU memory."""
    from contextlib import ExitStack
    from safetensors import safe_open

    mapping = _weight_map(scorer_path)
    keys = sorted(key for key in mapping if key.startswith("action_head."))
    head = copy.deepcopy(actor_head)
    with ExitStack() as stack:
        files = {
            name: stack.enter_context(
                safe_open(Path(scorer_path) / name, framework="pt", device="cpu")
            )
            for name in sorted({mapping[key] for key in keys})
        }
        state = {
            key.removeprefix("action_head."): files[mapping[key]].get_tensor(key)
            for key in keys
        }
        head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


@dataclass(frozen=True)
class RetrievedContext:
    task: str
    indices: tuple[int, ...]
    similarities: tuple[float, ...]
    actions: torch.Tensor
    backbone_output: BatchFeature
    source: str


def select_context(
    memory, task, query, k, device, dtype, *, indices=None, candidate_mask=None
):
    """Select once for both heads; optional explicit indices freeze the context.

    candidate_mask is an eligibility mask over memory rows. Future LODO code
    must construct it from verified demo provenance; row indices alone do not
    identify demos in the released memory.
    """
    if task not in memory:
        raise KeyError(f"Task not in context memory: {task!r}")
    entry = memory[task]
    keys = entry["keys"].to(device=device, dtype=dtype)
    if not isinstance(k, int) or not 1 <= k <= min(8, len(keys)):
        raise ValueError("context K must be in 1..8 and not exceed memory size")
    if query.shape != (1, keys.shape[1]):
        raise ValueError("context scorer accepts exactly one query observation")
    scores = (query @ keys.T)[0]
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite retrieval similarity")
    if candidate_mask is not None:
        mask = torch.as_tensor(candidate_mask, device=device)
        if mask.dtype != torch.bool or mask.shape != scores.shape:
            raise ValueError(
                "candidate mask must be boolean with one entry per memory row"
            )
        if int(mask.sum()) < k:
            raise ValueError("fewer than K eligible memory rows")
    else:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if indices is None:
        indices = scores.masked_fill(~mask, -torch.inf).topk(k).indices.tolist()
        source = "retriever_topk"
    else:
        indices = list(indices)
        if (
            len(indices) != k
            or len(set(indices)) != k
            or any(
                not isinstance(i, (int, np.integer))
                or isinstance(i, bool)
                or not 0 <= i < len(keys)
                for i in indices
            )
        ):
            raise ValueError(
                "explicit context indices must be K distinct valid memory rows"
            )
        if not bool(mask[indices].all()):
            raise ValueError("explicit context includes an ineligible memory row")
        source = "explicit_indices"
    indices = tuple(int(i) for i in indices)
    # CPU mmap memory keeps images and unselected VLM embeddings off the GPU.
    actions = entry["action"][list(indices)].to(device=device, dtype=dtype)
    visual = entry["backbone_features"][list(indices)].to(device=device, dtype=dtype)
    attention = entry["backbone_attention_mask"][list(indices)].to(device=device)
    return RetrievedContext(
        task,
        indices,
        tuple(scores[list(indices)].float().cpu().tolist()),
        actions,
        BatchFeature(
            data={"backbone_features": visual, "backbone_attention_mask": attention}
        ),
        source,
    )


def velocity_function(head, backbone_output, state, embodiment_ids, context):
    """Cache each head's own projections, without mutating shared raw features."""
    visual = head.process_backbone_output(BatchFeature(data=dict(backbone_output)))
    retrieved = head.process_backbone_output(
        BatchFeature(data=dict(context.backbone_output))
    )
    action_context = head.ret_action_encoder(context.actions)
    action_context += head.ret_position_embedding.weight.unsqueeze(0)
    state_features = head.state_encoder(state, embodiment_ids)
    future = head.future_tokens.weight.unsqueeze(0).expand(len(state), -1, -1)

    def velocity(actions, time):
        times = torch.full(
            (len(state),), int(time * head.num_timestep_buckets), device=state.device
        )
        features = head.action_encoder(actions, times, embodiment_ids)
        if head.config.add_pos_embed:
            positions = torch.arange(head.config.action_horizon, device=state.device)
            features = features + head.position_embedding(positions).unsqueeze(0)
        hidden = torch.cat([state_features, future, features], dim=1)
        output = head.model(
            hidden_states=hidden,
            encoder_hidden_states=visual.backbone_features,
            encoder_attention_mask=visual.backbone_attention_mask,
            ret_encoder_hidden_states=retrieved.backbone_features,
            ret_encoder_attention_mask=retrieved.backbone_attention_mask,
            ret_action_features=action_context,
            timestep=times,
        )
        return head.action_decoder(output, embodiment_ids)[
            :, -head.config.action_horizon :
        ]

    return velocity


class ContextVFDPolicy:
    """One actor plus an independent checkpoint's head, M=2 and B=1.

    Normalized scores sum squared disagreement over 16 steps and the 7 real
    LIBERO action dimensions. No Gaussian source noise or weight perturbation
    is added. Repeated calls with fixed observations/context are deterministic.
    """

    def __init__(
        self, policy, scorer_head, signature, *, denoising_steps=4, flow_times=None
    ):
        self.policy = policy
        self.scorer_head = scorer_head
        self.num_steps = denoising_steps
        self.flow_times = None if flow_times is None else list(flow_times)
        grid = flow_grid(denoising_steps, self.flow_times)
        actor = policy.vla
        if actor.training or actor.action_head.training or scorer_head.training:
            raise ValueError("actor and scorer must be in eval mode")
        for key in ("action_dim", "action_horizon", "num_timestep_buckets"):
            if getattr(actor.action_head, key) != getattr(scorer_head, key):
                raise ValueError(f"incompatible action heads: {key}")
        if actor.action_head.action_dim < 7:
            raise ValueError("LIBERO scoring requires seven real action dimensions")
        if actor.action_head.num_inference_timesteps != denoising_steps:
            raise ValueError("actor denoising steps differ from score configuration")
        self.signature = {
            **signature,
            "score_version": 1,
            "score": "context_conditioned_one_way_vfd_squared_l2",
            "interpretation": "diagnostic_adaptation_not_gaussian_source_KL",
            "ensemble_size": 2,
            "trajectory_count": 1,
            "trajectory_source": "mean_retrieved_normalized_actions",
            "trajectory_member": "actor",
            "context_shared_between_members": True,
            "denoising_steps": denoising_steps,
            "flow_times": [i / denoising_steps for i in grid],
            "quadrature_weights": list(grid.values()),
            "action_dimensions": list(range(7)),
            "action_horizon": actor.action_head.action_horizon,
            "action_coordinates": "checkpoint_normalized",
            "reduction": "sum_horizon_and_real_dimensions",
            "k": actor.k,
            "data_config": DATA_CONFIG,
        }
        self.score_id = _digest(self.signature)

    @classmethod
    def from_actor_policy(
        cls,
        policy,
        actor_path,
        scorer_path,
        retriever_path,
        memory_path,
        *,
        denoising_steps=4,
        flow_times=None,
    ):
        signature = validate_checkpoint_pair(actor_path, scorer_path, retriever_path)
        memory_path = Path(memory_path).resolve(strict=True)
        stat = memory_path.stat()
        signature["memory"] = {
            "path": str(memory_path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        signature["retriever_weights_sha256"] = _tensor_digest(retriever_path, "")
        signature["compute_dtype"] = str(policy.vla.action_head.dtype)
        repo = Path(__file__).resolve().parents[2]
        sources = [
            "gr00t/rag/context_vfd.py",
            "gr00t/rag/ravla.py",
            "gr00t/rag/retriever.py",
            "gr00t/model/transforms.py",
            "examples/Libero/custom_data_config.py",
            "gr00t/data/transform/base.py",
            "gr00t/data/transform/concat.py",
            "gr00t/data/transform/video.py",
            "gr00t/data/transform/state_action.py",
            "gr00t/model/backbone/eagle_backbone.py",
        ]
        sources.extend(
            str(p.relative_to(repo))
            for p in sorted(
                (repo / "gr00t/model/backbone/eagle2_hg_model").glob("*.json")
            )
        )
        signature["source_sha256"] = {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in sources
        }
        scorer = load_scorer_head(policy.vla.action_head, scorer_path)
        return cls(
            policy,
            scorer,
            signature,
            denoising_steps=denoising_steps,
            flow_times=flow_times,
        )

    def get_score_signature(self):
        return {"score_id": self.score_id, **copy.deepcopy(self.signature)}

    def _prepare_observation(self, observations):
        actor = self.policy.vla
        inputs = self.policy.collator([self.policy.transforms(observations.copy())])
        tasks = inputs.pop("tasks")
        if len(tasks) != 1:
            raise ValueError("context scorer requires a single unbatched observation")
        queries = actor.retriever[0](actor, inputs)
        backbone_inputs, action_inputs = actor.prepare_input(inputs)
        raw_visual = actor.backbone(backbone_inputs)
        return tasks[0], queries, action_inputs, raw_visual

    def _score_context(self, context, action_inputs, raw_visual):
        actor = self.policy.vla
        head = actor.action_head
        velocities = [
            velocity_function(
                member,
                raw_visual,
                action_inputs.state,
                action_inputs.embodiment_id,
                context,
            )
            for member in (head, self.scorer_head)
        ]
        initial = context.actions.reshape(
            1, actor.k, head.action_horizon, head.action_dim
        ).mean(1)
        terminal, scores, trace = integrate_context_vfd(
            *velocities,
            initial,
            self.num_steps,
            real_action_dim=7,
            flow_times=self.flow_times,
        )
        result = {
            "u_ctx": float(scores[0]),
            "score_id": self.score_id,
            "context": {
                "task": context.task,
                "indices": list(context.indices),
                "similarities": list(context.similarities),
                "source": context.source,
            },
            "flow_times": trace["flow_times"],
            "weights": trace["weights"].cpu().numpy(),
            "squared_l2": trace["squared_l2"][0].cpu().numpy(),
            "weighted_disagreement": trace["weighted_disagreement"][0].cpu().numpy(),
            "u_ctx_per_dimension": trace["per_dimension"][0].cpu().numpy(),
        }
        return terminal, result

    @torch.inference_mode()
    def get_action_with_uncertainty(
        self, observations, *, context_indices=None, candidate_mask=None
    ):
        actor = self.policy.vla
        head = actor.action_head
        task, queries, action_inputs, raw_visual = self._prepare_observation(
            observations
        )
        context = select_context(
            actor.memory,
            task,
            queries,
            actor.k,
            head.device,
            head.dtype,
            indices=context_indices,
            candidate_mask=candidate_mask,
        )
        terminal, result = self._score_context(context, action_inputs, raw_visual)
        result["action"] = self.policy.transforms.unapply(
            {"action": terminal[0].float().cpu()}
        )
        return result

    @torch.inference_mode()
    def get_action_with_context_diagnostics(self, observations):
        """Act with top-K; score two same-task distractors on identical observations.

        Each context uses RA-VLA's own retrieved-action initial condition and
        actor trajectory. Thus this compares the full context-conditioned policy,
        not attention alone. Distractors are proxies, not phase-mismatch labels.
        Random selection is derived from the query, independent of request order.
        """
        actor = self.policy.vla
        head = actor.action_head
        task, queries, inputs, visual = self._prepare_observation(observations)
        context = select_context(
            actor.memory, task, queries, actor.k, head.device, head.dtype
        )
        terminal, result = self._score_context(context, inputs, visual)
        result["action"] = self.policy.transforms.unapply(
            {"action": terminal[0].float().cpu()}
        )
        alternatives = diagnostic_context_indices(
            actor.memory[task]["keys"], queries, context.indices
        )
        result["context_diagnostics"] = {}
        for name, indices in alternatives.items():
            other = select_context(
                actor.memory,
                task,
                queries,
                actor.k,
                head.device,
                head.dtype,
                indices=indices,
            )
            other_terminal, other_result = self._score_context(other, inputs, visual)
            other_result["normalized_action_distance_to_executed"] = float(
                (other_terminal[..., :7].float() - terminal[..., :7].float())
                .square()
                .sum()
                .sqrt()
            )
            result["context_diagnostics"][name] = other_result
        result["diagnostics_definition"] = {
            "sources": ["random_same_task", "low_similarity_same_task"],
            "random_seed_rule": "sha256(query_float32_bytes), private_numpy_generator",
            "excluded_indices": list(context.indices),
            "selection_independent_of_uncertainty": True,
            "phase_alignment_ground_truth": False,
            "trajectory_rule": "each_context_uses_its_own_retrieved_action_source_and_actor_flow",
        }
        return result


def diagnostic_context_indices(keys, query, retrieved_indices):
    """Choose negative proxies without consulting VFD scores or global RNGs."""
    scores = (query @ keys.to(device=query.device, dtype=query.dtype).T)[0]
    k = len(retrieved_indices)
    eligible = np.asarray(
        [i for i in range(len(keys)) if i not in retrieved_indices], dtype=np.int64
    )
    if len(eligible) < k:
        raise ValueError("not enough disjoint same-task contexts for diagnostics")
    seed = int.from_bytes(
        hashlib.sha256(query.float().cpu().numpy().tobytes()).digest()[:8], "little"
    )
    random = np.random.default_rng(seed).choice(eligible, k, replace=False).tolist()
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[list(retrieved_indices)] = False
    low = scores.masked_fill(~mask, torch.inf).topk(k, largest=False).indices.tolist()
    return {"random_same_task": random, "low_similarity_same_task": low}
