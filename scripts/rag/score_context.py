"""Score raw, lossless LIBERO observation NPZs with two RA-VLA checkpoints.

This is a small offline scorer/debug entry point. It does not reconstruct
proprioception from MP4s, run a simulator, or fit calibration thresholds.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def json_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load_observation(path):
    """NPZ keys are the same raw modality keys accepted by VLAPolicy."""
    state = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    required = {
        "video.image",
        "video.wrist_image",
        "annotation.human.action.task_description",
    }
    required.update(f"state.{key}" for key in state)
    with np.load(path, allow_pickle=False) as file:
        missing = required - set(file.files)
        if missing:
            raise ValueError(
                f"Missing raw observation fields in {path}: {sorted(missing)}"
            )
        obs = {key: file[key].copy() for key in required}
    for key in ("video.image", "video.wrist_image"):
        if (
            obs[key].dtype != np.uint8
            or obs[key].ndim != 4
            or obs[key].shape[0] != 1
            or obs[key].shape[-1] != 3
        ):
            raise ValueError(f"{key} must be uint8 [1, H, W, 3]")
    for key in state:
        shape = (1, 2 if key == "gripper" else 1)
        if (
            obs[f"state.{key}"].shape != shape
            or not np.isfinite(obs[f"state.{key}"]).all()
        ):
            raise ValueError(f"state.{key} must be finite with shape {shape}")
    text = obs["annotation.human.action.task_description"]
    if text.shape != (1,) or text.dtype.kind != "U" or not str(text[0]):
        raise ValueError("task description must be a nonempty Unicode array [1]")
    obs["annotation.human.action.task_description"] = text.tolist()
    return obs


def load_monitor(args):
    import torch
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import load_data_config
    from gr00t.rag.context_vfd import ContextVFDPolicy, DATA_CONFIG, EMBODIMENT
    from gr00t.rag.policy import VLAPolicy
    from gr00t.rag.ravla import RAVLA, RAVLADataCollator
    from gr00t.rag.retriever import RetrieverWrapperWithHead

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real checkpoint scoring requires an available CUDA device")
    config = load_data_config(DATA_CONFIG)
    transforms = config.transform()
    metadata = torch.load(
        args.actor / "metadata.pt", map_location="cpu", weights_only=False
    )
    transforms.set_metadata(metadata[EMBODIMENT])
    transforms.eval()
    actor = RAVLA.from_pretrained(
        str(args.actor), torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    retriever = RetrieverWrapperWithHead.from_pretrained(
        str(args.retriever),
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    memory = torch.load(args.memory, map_location="cpu", mmap=True, weights_only=True)
    for entry in memory.values():
        entry["keys"] = entry["keys"].to(device=device, dtype=torch.bfloat16)
    actor.set_retrieval_components(retriever, memory, args.k)
    actor.action_head.num_inference_timesteps = args.denoising_steps
    actor.compute_dtype = actor.config.compute_dtype = "bfloat16"
    policy = VLAPolicy(
        config.modality_config(),
        transforms,
        EmbodimentTag(EMBODIMENT),
        RAVLADataCollator(),
        actor,
    )
    return ContextVFDPolicy.from_actor_policy(
        policy,
        args.actor,
        args.scorer,
        args.retriever,
        args.memory,
        denoising_steps=args.denoising_steps,
        flow_times=args.flow_times,
    )


def check_parity(monitor, observation, result):
    """Debug only: run the original actor path against the same raw input."""
    import torch

    reference = monitor.policy.get_action(observation)
    deltas = []
    for key, value in reference.items():
        left = np.asarray(value)
        right = np.asarray(result["action"][key])
        if not np.array_equal(left, right):
            raise AssertionError(
                f"monitored action differs from original actor for {key}"
            )
        deltas.append(float(np.max(np.abs(left - right))))
    repeated = monitor.get_action_with_uncertainty(observation)
    if result["u_ctx"] != repeated["u_ctx"] or result["context"] != repeated["context"]:
        raise AssertionError("repeated context score is not deterministic")
    # The same field compared to itself must give exactly zero.
    scorer = monitor.scorer_head
    try:
        monitor.scorer_head = monitor.policy.vla.action_head
        identical = monitor.get_action_with_uncertainty(observation)
    finally:
        monitor.scorer_head = scorer
    if identical["u_ctx"] != 0:
        raise AssertionError("identical-head disagreement is nonzero")
    for key in result["action"]:
        torch.testing.assert_close(
            torch.as_tensor(result["action"][key]),
            torch.as_tensor(identical["action"][key]),
            rtol=0,
            atol=0,
        )
    return {
        "original_actor_max_abs_delta": max(deltas),
        "repeat_exact": True,
        "identical_head_score": 0.0,
    }


def write_result(path, value):
    """Publish complete JSON atomically, refusing existing/concurrent outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".ctx_", delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2, default=json_value, allow_nan=False)
            file.write("\n")
            file.close()
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument(
        "--retriever", type=Path, default=Path("ra-vla-official-assets/libero_gos")
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=Path("ra-vla-official-assets/libero_gos/memory_l.pt"),
    )
    parser.add_argument("--observations", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--flow-times", nargs="+", type=float)
    parser.add_argument("--context-indices", nargs="+", type=int)
    parser.add_argument("--check-parity", action="store_true")
    args = parser.parse_args()
    from gr00t.rag.context_vfd import flow_grid
    from gr00t.rag.utils import seed_everything

    flow_grid(args.denoising_steps, args.flow_times)
    if not 1 <= args.k <= 8:
        parser.error("K must be in 1..8")
    if args.check_parity and args.context_indices is not None:
        parser.error(
            "parity requires the normal retriever, without explicit context indices"
        )
    observations = [load_observation(p) for p in args.observations]
    if args.output.exists():
        raise FileExistsError(args.output)
    seed_everything(0)
    monitor = load_monitor(args)
    results = []
    for path, obs in zip(args.observations, observations):
        result = monitor.get_action_with_uncertainty(
            obs, context_indices=args.context_indices
        )
        if args.check_parity:
            result["parity_checks"] = check_parity(monitor, obs, result)
        results.append({"observation_path": str(path.resolve()), **result})
        print(
            f"{path.name}: U_ctx={result['u_ctx']:.6g}; context={result['context']['indices']}",
            flush=True,
        )
    write_result(
        args.output,
        {"score_signature": monitor.get_score_signature(), "results": results},
    )


if __name__ == "__main__":
    main()
