"""Re-encode retrieval keys while preserving every other tensor in a memory buffer."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from scripts.rag.precompute_retrieval import atomic_json, model_stamp, stamp


def memory_inputs(task, pixels, transform, collator):
    """Use normal image/text token layout, then insert the exact cached pixels.

    The buffer already contains normalized, cropped images. Running the full
    video transforms again would crop them twice. Placeholder images only build
    the two-camera prompt/layout; their processed pixel values are discarded.
    """
    if pixels.ndim != 5 or tuple(pixels.shape[1:3]) != (2, 3):
        raise ValueError(f"Expected [N, 2, 3, H, W] cached pixels, got {pixels.shape}")
    count, _, _, height, width = pixels.shape
    feature = transform(
        {
            "video": np.zeros((1, 2, height, width, 3), dtype=np.uint8),
            "annotation.human.action.task_description": [task],
        }
    )
    inputs = collator([feature] * count)
    cached = pixels.flatten(0, 1)
    if inputs["eagle_pixel_values"].shape != cached.shape:
        raise ValueError("Cached images disagree with processor's image/tile layout")
    inputs["eagle_pixel_values"] = cached
    return inputs


def verify_payload(original, rewritten):
    if original.keys() != rewritten.keys():
        raise ValueError("Task sets changed")
    count = 0
    for task, payload in original.items():
        if payload.keys() != rewritten[task].keys():
            raise ValueError(f"Payload fields changed for {task}")
        for name, tensor in payload.items():
            replacement = rewritten[task][name]
            if tensor.shape != replacement.shape or tensor.dtype != replacement.dtype:
                raise ValueError(f"Shape/dtype changed: {task}/{name}")
            if name != "keys":
                if not torch.equal(
                    tensor.contiguous().view(torch.uint8), replacement.contiguous().view(torch.uint8)
                ):
                    raise ValueError(f"Payload changed: {task}/{name}")
                count += tensor.numel() * tensor.element_size()
    return count


def save_memory(original, rewritten, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch's zip writer rejects some hidden extensionless temporary names.
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix="pending_rekey_", suffix=".pt", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(rewritten, temporary)
        reloaded = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
        unchanged_bytes = verify_payload(original, reloaded)
        for task in rewritten:
            if not torch.equal(rewritten[task]["keys"], reloaded[task]["keys"]):
                raise ValueError(f"Saved keys disagree for {task}")
        os.replace(temporary, output)
        return unchanged_bytes
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-memory", type=Path, required=True)
    parser.add_argument("--output-memory", type=Path, required=True)
    parser.add_argument("--retriever-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Policy used for eval; supplies the frozen visual/text encoder",
    )
    parser.add_argument(
        "--reference-retriever-path",
        type=Path,
        help="Original retriever: validate reconstructed inputs against source keys",
    )
    parser.add_argument("--reference-min-cosine", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cuda < 0 or not -1 <= args.reference_min_cosine <= 1:
        parser.error("Invalid batch size, GPU index or cosine threshold")
    if args.source_memory.resolve() == args.output_memory.resolve():
        parser.error("Output must differ from the source buffer")
    if args.output_memory.exists():
        parser.error("Output already exists; select a new output path")
    if not torch.cuda.is_available():
        parser.error("Run on a GPU allocation")

    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.transforms import DefaultDataCollator
    from gr00t.rag.ravla import RAVLA
    from gr00t.rag.retriever import RetrieverWrapperWithHead

    started = time.perf_counter()
    device = f"cuda:{args.cuda}"
    torch.manual_seed(0)
    source = torch.load(args.source_memory, map_location="cpu", mmap=True, weights_only=True)
    transform = LiberoDataConfig().transform().transforms[-1]
    metadata = torch.load(args.checkpoint_path / "metadata.pt", map_location="cpu", weights_only=False)
    transform.set_metadata(metadata["new_embodiment"])
    transform.eval()
    collator = DefaultDataCollator()
    vla = RAVLA.from_pretrained(str(args.checkpoint_path), torch_dtype=torch.bfloat16, device_map=device).eval()
    vla.requires_grad_(False)
    retriever = RetrieverWrapperWithHead.from_pretrained(
        str(args.retriever_path), torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    reference = None
    if args.reference_retriever_path:
        reference = RetrieverWrapperWithHead.from_pretrained(
            str(args.reference_retriever_path), torch_dtype=torch.bfloat16, device_map=device
        ).eval()
    report = {
        "source_memory": stamp(args.source_memory),
        "retriever": model_stamp(args.retriever_path),
        "policy": model_stamp(args.checkpoint_path),
        "reference_retriever": model_stamp(args.reference_retriever_path) if reference else None,
        "batch_size": args.batch_size,
        "tasks": [],
    }
    rewritten = {}
    with torch.inference_mode():
        for task, payload in source.items():
            keys = []
            entry = {"task": task, "candidates": len(payload["keys"])}
            for start in range(0, entry["candidates"], args.batch_size):
                pixels = payload["pixel_values"][start : start + args.batch_size]
                inputs = memory_inputs(task, pixels, transform, collator)
                if start == 0 and reference is not None:
                    expected = payload["keys"][: len(pixels)].float()
                    actual = reference(vla, inputs).float().cpu()
                    cosine = cosine_similarity(actual, expected, dim=-1)
                    entry["reference_min_cosine"] = float(cosine.min())
                    entry["reference_mean_cosine"] = float(cosine.mean())
                    entry["reference_rmse"] = float((actual - expected).square().mean().sqrt())
                    if cosine.min() < args.reference_min_cosine:
                        raise ValueError(
                            f"Reconstructed input check failed for {task}: min cosine {float(cosine.min())}"
                        )
                encoded = retriever(vla, inputs).cpu()
                if not torch.isfinite(encoded).all():
                    raise ValueError(f"Non-finite keys for {task}")
                keys.append(encoded)
            rewritten[task] = {**payload, "keys": torch.cat(keys).to(payload["keys"].dtype)}
            entry["new_key_norm_mean"] = float(rewritten[task]["keys"].float().norm(dim=-1).mean())
            report["tasks"].append(entry)
            print(json.dumps(entry), flush=True)

    report["unchanged_payload_bytes"] = save_memory(source, rewritten, args.output_memory)
    report.update(
        {
            "payload_bitwise_equal": True,
            "only_keys_changed": True,
            "output_memory": stamp(args.output_memory),
            "seconds": time.perf_counter() - started,
            "total_candidates": sum(item["candidates"] for item in report["tasks"]),
        }
    )
    atomic_json(args.output_memory.with_suffix(".json"), report)
    print(
        f"Saved {report['total_candidates']} keys; all non-key tensors verified bitwise equal: {args.output_memory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
