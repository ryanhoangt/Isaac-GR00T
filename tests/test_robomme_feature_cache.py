"""Cache source alignment, crash/resume integrity and frozen vision extraction."""

import copy
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from examples.RoboMME.feature_cache import (
    build_feature_cache,
    cache_plan,
    load_cache_config,
    load_cached_episode,
)
from examples.RoboMME.memory_config import load_memory_config, prepare_frame_sampling
from examples.RoboMME.visual_features import (
    FrozenMemoryVisionEncoder,
    load_frozen_eagle_vision,
    pool_visual_tokens,
    preprocess_memory_images,
    vision_weights_sha256,
)


def image_value(level):
    stream = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), level, np.uint8)).save(stream, format="PNG")
    return {"bytes": stream.getvalue(), "path": None}


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    episodes = []
    for episode_id, flags in ((3, [True, True, False, False, False]), (1001, [False, False])):
        length = len(flags)
        data = pd.DataFrame(
            {
                "image": [image_value(index * 20) for index in range(length)],
                "episode_index": [episode_id] * length,
                "frame_index": list(range(length)),
                "is_demo": flags,
                "exec_start_idx": [sum(flags)] * length,
                "state": [np.array([index] * 8, np.float32) for index in range(length)],
                "actions": [np.array([index + 10] * 8, np.float32) for index in range(length)],
            }
        )
        path = source / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        episodes.append({"episode_index": episode_id, "length": length})
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {key: {"dtype": "image", "shape": [32, 32, 3]} for key in ("image", "wrist_image")},
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (source / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"example"}\n')
    return source


def config():
    result = load_memory_config()
    result["memory_feature"]["img"]["input_dim"] = 4
    return result


class FakeEncoder:
    def __init__(self, fail_on_call=None):
        self.signature = {"pool_type": "mean", "token_per_image": 16, "image_dim": 4, "weights": "fake-v1"}
        self.calls = []
        self.fail_on_call = fail_on_call

    def encode(self, images):
        self.calls.append(len(images))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("interrupted encoder")
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.broadcast_to(values[:, None, None], (len(images), 16, 4)).copy()


def test_cache_includes_demo_and_execution_with_source_ids(source, tmp_path):
    plan = cache_plan(source, config())
    assert plan["total_frames"] == 7 and plan["feature_bytes"] == 7 * 16 * 4 * 2
    assert [row["source_episode_index"] for row in plan["source_episodes"]] == [3, 1001]
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "cache"
    manifest = build_feature_cache(plan, output, FakeEncoder(), batch_size=2)
    assert manifest["status"] == "complete" and manifest["completed_frames"] == 7
    features, record = load_cached_episode(output, 3)
    assert isinstance(features, np.memmap) and features.dtype == np.float16
    assert not features.flags.writeable
    np.testing.assert_array_equal(features[:, 0, 0], [0, 20, 40, 60, 80])
    assert record["source"]["exec_start_idx"] == 2
    assert record["source"]["execution_frames"] == 3
    assert record["verification"]["passed"]
    assert record["verification"]["source_frame_indices"] == [0, 2, 4]
    assert yaml.safe_load((output / "memory_config.yaml").read_text()) == config()
    # At the first execution sample the two demo frames remain accessible.
    memory = prepare_frame_sampling(features, 2, config())
    np.testing.assert_array_equal(memory["static_image_emb"][::16, 0][:3], [0, 20, 40])
    assert memory["static_mask"].sum() == 48
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_resume_reuses_completed_episodes_without_encoder_calls(source, tmp_path):
    plan = cache_plan(source, config())
    output = tmp_path / "cache"
    build_feature_cache(plan, output, FakeEncoder(), batch_size=2)
    encoder = FakeEncoder(fail_on_call=1)
    # Downstream sampling configuration can change without rebuilding visual tokens.
    changed = config()
    changed["budget"] = 256
    build_feature_cache(cache_plan(source, changed), output, encoder, batch_size=4, resume=True)
    assert not encoder.calls
    with pytest.raises(FileExistsError):
        build_feature_cache(plan, output, encoder)


def test_interruption_only_publishes_complete_episodes(source, tmp_path):
    output = tmp_path / "cache"
    plan = cache_plan(source, config())
    # Three extraction batches + three verification batches for episode 3.
    with pytest.raises(RuntimeError, match="interrupted"):
        build_feature_cache(plan, output, FakeEncoder(fail_on_call=7), batch_size=2)
    assert (output / "episode_000003/record.json").is_file()
    assert not (output / "episode_001001").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "building"
    encoder = FakeEncoder()
    result = build_feature_cache(plan, output, encoder, batch_size=2, resume=True)
    assert result["status"] == "complete" and encoder.calls == [2, 2]


def test_corrupt_cache_and_wrong_encoder_are_rejected(source, tmp_path):
    output = tmp_path / "cache"
    plan = cache_plan(source, config())
    build_feature_cache(plan, output, FakeEncoder())
    encoder = FakeEncoder()
    encoder.signature["weights"] = "different"
    with pytest.raises(ValueError, match="encoder weights"):
        build_feature_cache(plan, output, encoder, resume=True)
    path = output / "episode_000003/features.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        build_feature_cache(plan, output, FakeEncoder(), resume=True)


def test_verification_failure_does_not_publish_episode(source, tmp_path):
    class ChangedEncoder(FakeEncoder):
        def encode(self, images):
            encoded = super().encode(images)
            return encoded + (1 if len(self.calls) > 1 else 0)

    output = tmp_path / "cache"
    plan = cache_plan(source, config(), episode_indices=[3])
    with pytest.raises(ValueError, match="Cached/direct feature mismatch"):
        build_feature_cache(plan, output, ChangedEncoder(), batch_size=16)
    assert not (output / "episode_000003").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "building"


def test_local_image_changes_invalidate_resume_without_parquet_changes(source, tmp_path):
    path = source / "data/chunk-000/episode_000003.parquet"
    data = pd.read_parquet(path)
    asset = source / "front.png"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(asset)
    data["image"] = [{"bytes": None, "path": "front.png"}] * len(data)
    data.to_parquet(path, index=False)
    output = tmp_path / "cache"
    plan = cache_plan(source, config(), episode_indices=[3])
    build_feature_cache(plan, output, FakeEncoder())
    Image.fromarray(np.ones((32, 32, 3), np.uint8)).save(asset)
    with pytest.raises(ValueError, match="provenance differs"):
        build_feature_cache(plan, output, FakeEncoder(), resume=True)


def test_changed_source_even_when_stat_is_preserved_rejected(source, tmp_path):
    import os

    output = tmp_path / "cache"
    plan = cache_plan(source, config())
    build_feature_cache(plan, output, FakeEncoder())
    path = source / "data/chunk-000/episode_000003.parquet"
    stat = path.stat()
    data = pd.read_parquet(path)
    data["image"] = [image_value(42)] * len(data)
    data.to_parquet(path, index=False)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Replan to accept legitimate changed file size; contents still invalidate cache.
    with pytest.raises(ValueError):
        build_feature_cache(cache_plan(source, config()), output, FakeEncoder(), resume=True)


@pytest.mark.parametrize(
    "flags,start",
    [([True, False, True, False, False], 1), ([True] * 5, 5), ([True, True, False, False, False], 0)],
)
def test_invalid_demo_boundaries(source, tmp_path, flags, start):
    path = source / "data/chunk-000/episode_000003.parquet"
    data = pd.read_parquet(path)
    data["is_demo"] = flags
    data["exec_start_idx"] = start
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="execution|demo prefix"):
        build_feature_cache(cache_plan(source, config()), tmp_path / "cache", FakeEncoder())


def test_dry_run_does_not_decode_images_or_load_model(source, tmp_path, monkeypatch, capsys):
    from scripts.cache_robomme_visual_features import main

    monkeypatch.setattr("examples.RoboMME.feature_cache.decode_image", lambda *args: pytest.fail("decoded"))
    output = tmp_path / "cache"
    assert (
        main(["--source", str(source), "--output", str(output), "--model-path", "/missing/model", "--dry-run"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["total_frames"] == 7
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value", [("batch_size", 0), ("storage_dtype", "bfloat16"), ("verify", "yes"), ("compute_dtype", "fp8")]
)
def test_cache_config_rejects_invalid_values(tmp_path, field, value):
    config = load_cache_config()
    config[field] = value
    path = tmp_path / "cache.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_cache_config(path)


def test_preprocessing_rgb_normalization_aspect_and_padding():
    images = np.zeros((2, 112, 224, 3), np.uint8)
    images[0, :, :, 0] = 255
    pixels = preprocess_memory_images(images)
    assert pixels.shape == (2, 3, 224, 224) and pixels.dtype == torch.float32
    assert (pixels[:, :, :56] == -1).all() and (pixels[:, :, 168:] == -1).all()
    assert (pixels[0, 0, 56:168] == 1).all()
    assert (pixels[0, 1:] == -1).all()
    assert (pixels[1] == -1).all()
    with pytest.raises(ValueError, match="uint8"):
        preprocess_memory_images(images.astype(np.float32))


@pytest.mark.parametrize("method", ["mean", "max"])
def test_pooling_matches_nonoverlapping_spatial_cells(method):
    tokens = torch.arange(16 * 16 * 2).reshape(1, 256, 2).float()
    pooled = pool_visual_tokens(tokens, method)
    grid = tokens.numpy().reshape(16, 16, 2)
    expected = []
    for row in range(4):
        for col in range(4):
            cell = grid[row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4]
            expected.append(cell.mean((0, 1)) if method == "mean" else cell.max((0, 1)))
    np.testing.assert_array_equal(pooled.numpy()[0], expected)


def tiny_eagle():
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

    config = Eagle2_5_VLConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 224,
            "patch_size": 14,
        },
        text_config={
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
        },
        use_pixel_shuffle=False,
        mlp_connector_layers=1,
        force_image_size=224,
        select_layer=-1,
        _attn_implementation="sdpa",
    )
    config.vision_config._attn_implementation = "sdpa"
    return (
        Eagle2_5_VLForConditionalGeneration(
            config, vision_model=SiglipVisionModel(config.vision_config), language_model=nn.Identity()
        )
        .eval()
        .requires_grad_(False)
    )


def test_selective_loader_uses_existing_eagle_path_and_materializes_buffers(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    eagle = tiny_eagle()
    (tmp_path / "config.json").write_text('{"model_type":"gr00t_n1_5"}')
    state = {"backbone.eagle_model." + key: value for key, value in eagle.state_dict().items()}
    # A language tensor is present in the checkpoint but must not be loaded.
    state["backbone.eagle_model.language_model.unused"] = torch.ones(1)
    save_file(state, tmp_path / "model.safetensors")
    model_config = copy.deepcopy(eagle.config)
    monkeypatch.setattr(
        "gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl.Eagle2_5_VLConfig",
        lambda **kwargs: copy.deepcopy(model_config),
    )
    loaded = load_frozen_eagle_vision(tmp_path, "sdpa")
    assert not any(buffer.is_meta for buffer in loaded.buffers())
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert vision_weights_sha256(loaded) == vision_weights_sha256(eagle)
    images = np.full((2, 32, 32, 3), 83, np.uint8)
    original = FrozenMemoryVisionEncoder(eagle, config(), device="cpu", compute_dtype="float32")
    restored = FrozenMemoryVisionEncoder(loaded, config(), device="cpu", compute_dtype="float32")
    np.testing.assert_array_equal(original.encode(images), restored.encode(images))
    assert original.signature == restored.signature
    state.pop(next(key for key in state if "vision_model" in key))
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_frozen_eagle_vision(tmp_path, "sdpa")


def test_live_encoder_rejects_trainable_vision_and_fingerprint_changes():
    eagle = tiny_eagle()
    before = vision_weights_sha256(eagle)
    with torch.no_grad():
        eagle.mlp1[0].bias.add_(1)
    assert vision_weights_sha256(eagle) != before
    eagle.vision_model.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenMemoryVisionEncoder(eagle, config(), device="cpu", compute_dtype="float32")
