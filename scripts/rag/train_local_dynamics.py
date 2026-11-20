"""Fit and validate local chunk dynamics from existing GOS demonstrations."""

import json
from pathlib import Path

import numpy as np
import torch

from gr00t.rag.endpoint_data import collect_transitions
from gr00t.rag.local_dynamics import DeltaMLP, DynamicsEnsemble, PolicyCodec


def statistics(error):
    error = np.asarray(error, dtype=float)
    return dict(
        mean_m=float(error.mean()),
        median_m=float(np.median(error)),
        p95_m=float(np.quantile(error, 0.95)),
        max_m=float(error.max()),
    )


def quaternion_axis_angle(quat):
    q = torch.as_tensor(quat, dtype=torch.float32)
    w = q[..., 3].clamp(-1, 1)
    den = (1 - w.square()).clamp_min(0).sqrt()
    scale = torch.where(den > 1e-6, 2 * w.acos() / den.clamp_min(1e-8), torch.zeros_like(w))
    return q[..., :3] * scale[..., None]


@torch.no_grad()
def observed_policy_errors(ensemble, codec, root, device):
    rows = []
    for path in sorted(Path(root).glob("*/*/trace.npz")):
        episode = json.loads((path.parent / "episode.json").read_text())
        if episode.get("execute_steps") != 16:
            continue
        with np.load(path, allow_pickle=False) as trace:
            state = (
                torch.cat(
                    (
                        torch.as_tensor(trace["ee"][0]),
                        quaternion_axis_angle(trace["quat"][0]),
                        torch.as_tensor(trace["gripper"][0]),
                    )
                )
                .float()
                .to(device)[None]
            )
            action = torch.as_tensor(trace["actions"], dtype=torch.float32, device=device)[None]
            # The saved trace is in deployed LIBERO coordinates, whereas the
            # demo action has a [0,1] gripper coordinate inverted at deployment.
            action[..., 6] = (1 - action[..., 6]) / 2
            target = torch.as_tensor(trace["ee"][-1] - trace["ee"][0], dtype=torch.float32, device=device)[None]
        prediction, spread = ensemble(codec.state(state), codec.actions(action))
        rows.append(
            dict(
                case=path.parent.name,
                mode=episode["mode"],
                error_m=float((prediction - target).norm()),
                std_m=float(spread),
            )
        )
    if not rows:
        raise ValueError("No observation-only policy traces for dynamics validation")
    return rows


def fit(args, recipe, output, run=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    metadata = torch.load(args.checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
    codec = PolicyCodec.from_metadata(metadata).to(device)
    if not torch.allclose(codec.action_min[-1], torch.tensor(0.0, device=device)) or not torch.allclose(
        codec.action_max[-1], torch.tensor(1.0, device=device)
    ):
        raise ValueError("This pilot requires the existing [0,1] demonstration gripper convention")
    train, train_provenance = collect_transitions(recipe, "train")
    validation, val_provenance = collect_transitions(recipe, "validation")
    assert not set((d, e) for d, e, _ in train_provenance) & set((d, e) for d, e, _ in val_provenance)
    s, a, y = (v.to(device) for v in train)
    vs, va, vy = (v.to(device) for v in validation)
    x = DeltaMLP.inputs(codec.state(s), codec.actions(a))
    vx = DeltaMLP.inputs(codec.state(vs), codec.actions(va))
    # Train-only affine ridge baseline. It is reported, not chosen by repeatedly
    # tuning against the held-out demos.
    xmean = x.mean(0)
    xscale = (x - xmean).abs().mean(0).clamp_min(1e-3)
    design = torch.cat(((x - xmean) / xscale, torch.ones(len(x), 1, device=device)), dim=-1).double()
    regularizer = torch.eye(design.shape[1], device=device, dtype=torch.float64) * 1e-2
    regularizer[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ y.double())
    vdesign = torch.cat(((vx - xmean) / xscale, torch.ones(len(vx), 1, device=device)), dim=-1).double()
    ridge_error = (vdesign @ weights - vy).float().norm(dim=-1).cpu().numpy()
    torch.manual_seed(args.seed)
    ensemble = DynamicsEnsemble(args.ensemble, args.hidden, args.spectral).to(device)
    for number, member in enumerate(ensemble.members):
        member.fit_normalization(x, y)
        optimizer = torch.optim.Adam(member.parameters(), lr=1e-3, weight_decay=1e-5)
        generator = torch.Generator(device=device).manual_seed(args.seed + number)
        for epoch in range(args.dynamics_epochs):
            order = torch.randperm(len(x), generator=generator, device=device)
            total = 0.0
            for start in range(0, len(x), 512):
                indices = order[start : start + 512]
                prediction = member(x[indices])
                loss = ((prediction - y[indices]) / member.output_scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite dynamics training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                metrics = dict(member=number, epoch=epoch + 1, loss=total / len(x))
                print("DYNAMICS", json.dumps(metrics), flush=True)
                if run is not None:
                    run.log({f"dynamics/member{number}/loss": metrics["loss"], "dynamics/epoch": epoch + 1})
    ensemble.freeze()
    with torch.no_grad():
        prediction, spread = ensemble(codec.state(vs), codec.actions(va))
        errors = (prediction - vy).norm(dim=-1).cpu().numpy()
        train_prediction, train_spread = ensemble(codec.state(s), codec.actions(a))
    by_task = []
    for dataset in sorted({p[0] for p in val_provenance}):
        mask = np.asarray([p[0] == dataset for p in val_provenance])
        by_task.append(dict(dataset=dataset, **statistics(errors[mask])))
    policy = observed_policy_errors(ensemble, codec, args.policy_reference, device)
    goal_errors = [row["error_m"] for row in policy if row["mode"] == "goal"]
    if len(goal_errors) != 6:
        raise ValueError("Expected the existing six fixed-goal policy windows")
    general = statistics(errors)
    ready = (
        all(row["p95_m"] <= args.max_p95_m and row["mean_m"] <= args.max_mean_m for row in by_task)
        and max(goal_errors) <= args.max_policy_error_m
    )
    # Empirical spread filter; not a CCIL theoretical error bound.
    std_limit = float(torch.quantile(train_spread.float(), 0.99))
    report = dict(
        status="completed",
        ready_for_endpoint=ready,
        training_transitions=len(x),
        validation_transitions=len(vx),
        architecture=dict(
            input_dim=120,
            hidden=[args.hidden, args.hidden],
            output="EE delta xyz in meters",
            ensemble=args.ensemble,
            spectral=args.spectral,
        ),
        validation=general,
        per_task=by_task,
        ridge_validation=statistics(ridge_error),
        zero_motion_validation=statistics(vy.norm(dim=-1).cpu().numpy()),
        observed_policy_errors=policy,
        policy_goal_validation=statistics(goal_errors),
        training_std_p99_m=std_limit,
        gates=dict(
            max_mean_m=args.max_mean_m, max_p95_m=args.max_p95_m, max_policy_error_m=args.max_policy_error_m
        ),
        caveat="Empirical reliability filters, not guarantees on unseen actions or correct dynamics gradients; no CCIL labels generated.",
    )
    # Model sensitivity with respect to normalized policy state/actions.
    selected = vx[: min(len(vx), 32)].detach().requires_grad_(True)
    jacobian = torch.func.vmap(torch.func.jacrev(lambda item: ensemble.members[0](item[None])[0]))(selected)
    report["model_jacobian_spectral_mean"] = float(torch.linalg.matrix_norm(jacobian, ord=2).mean())
    torch.save(
        dict(
            ensemble=ensemble.cpu().state_dict(),
            codec=codec.cpu().state_dict(),
            hidden=args.hidden,
            count=args.ensemble,
            spectral=args.spectral,
            std_limit=std_limit,
            report=report,
            train_provenance=train_provenance,
            validation_provenance=val_provenance,
        ),
        output / "dynamics.pt",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DYNAMICS_VALIDATION", json.dumps(report), flush=True)
    if run is not None:
        run.log(
            {
                "dynamics/validation_p95_m": general["p95_m"],
                "dynamics/ready": int(ready),
                "dynamics/policy_goal_max_error_m": max(goal_errors),
            }
        )
    return report


def load(path, device="cuda:0"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not saved["report"]["ready_for_endpoint"]:
        raise ValueError("Dynamics validation did not pass")
    ensemble = DynamicsEnsemble(saved["count"], saved["hidden"], saved["spectral"])
    ensemble.load_state_dict(saved["ensemble"])
    codec_state = saved["codec"]
    codec = PolicyCodec(
        codec_state["state_min"], codec_state["state_max"], codec_state["action_min"], codec_state["action_max"]
    )
    return ensemble.to(device).freeze(), codec.to(device), saved["std_limit"]
