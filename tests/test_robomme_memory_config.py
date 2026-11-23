"""FrameSamp contracts: source time, causal selection, padding and configuration."""

import copy
import math

import numpy as np
import pytest
import yaml

from examples.RoboMME.memory_config import (
    DEFAULT_MEMORY_CONFIG,
    frame_sampling_indices,
    load_memory_config,
    position_embedding_3d,
    prepare_frame_sampling,
    validate_memory_config,
)


def test_yaml_defaults_and_custom_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_memory_config()  # bundled default must not depend on cwd
    assert config["budget"] == 512 and config["token_per_image"] == 16
    assert config["pool_type"] == "mean" and config["num_views"] == 1
    assert config["attention"]["num_heads"] * config["attention"]["head_dim"] == 1536
    config["budget"] = 256
    config["pool_type"] = "max"
    path = tmp_path / "memory.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_memory_config(path) == config
    assert DEFAULT_MEMORY_CONFIG.is_absolute()


@pytest.mark.parametrize(
    "step,capacity,expected",
    [
        (0, 32, [0]),
        (3, 32, [0, 1, 2, 3]),
        (31, 32, list(range(32))),
        (32, 4, [0, 10, 21, 32]),
        (100, 4, [0, 33, 66, 100]),
        (4095, 4, [0, 1365, 2730, 4095]),
    ],
)
def test_sampler_matches_upstream_even_sampling(step, capacity, expected):
    result = frame_sampling_indices(step, capacity)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int32


@pytest.mark.parametrize("step,capacity", [(-1, 32), (1.5, 32), (True, 32), (3, 0), (3, 2.5)])
def test_reject_invalid_sampling_indices(step, capacity):
    with pytest.raises(ValueError):
        frame_sampling_indices(step, capacity)


def test_position_features_match_source_axis_order_and_coordinates():
    # Independent scalar oracle for source PosEmb3D(dim=12), including an
    # absolute frame index that differs from its sampled sequence slot.
    result = position_embedding_3d([0, 83], dim=12)
    expected = []
    for timestep in (0, 83):
        frame = []
        for row in range(4):
            for col in range(4):
                coordinates = (timestep, 4 * row + 2, 4 * col + 2)
                token = []
                for coordinate, base in zip(coordinates, (10000, 1000, 1000)):
                    token.extend(
                        [
                            math.sin(coordinate),
                            math.sin(coordinate / base),
                            math.cos(coordinate),
                            math.cos(coordinate / base),
                        ]
                    )
                frame.append(token)
        expected.append(frame)
    np.testing.assert_allclose(result, expected, atol=2e-7)
    assert result.dtype == np.float32 and result.shape == (2, 16, 12)
    assert position_embedding_3d([], dim=768).shape == (0, 16, 768)


def small_config():
    config = load_memory_config()
    config["budget"] = 64
    config["memory_feature"]["img"]["input_dim"] = 2
    config["memory_feature"]["pos"]["input_dim"] = 12
    return config


def test_short_prefix_right_padding_and_independent_output():
    config = small_config()
    features = np.arange(10 * 16 * 2, dtype=np.float16).reshape(10, 16, 2)
    features[3:] = np.nan  # future observations must never be read
    result = prepare_frame_sampling(features, 2, config)
    np.testing.assert_array_equal(result["static_image_emb"][:48], features[:3].reshape(48, 2))
    assert not result["static_image_emb"][48:].any()
    assert not result["static_pos_emb"][48:].any()
    np.testing.assert_array_equal(result["static_mask"], [True] * 48 + [False] * 16)
    assert result["static_mask"].dtype == np.bool_
    assert result["static_pos_emb"].shape == (64, 12)
    result["static_image_emb"][0] = -1
    assert features[0, 0, 0] == 0


def test_long_prefix_uses_original_times_and_memmap(tmp_path):
    path = tmp_path / "features.npy"
    features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(102, 16, 2))
    features[:] = np.arange(102)[:, None, None]
    features[101] = np.nan
    features.flush()
    features = np.load(path, mmap_mode="r")
    result = prepare_frame_sampling(features, 100, small_config())
    assert result["static_mask"].all()
    np.testing.assert_array_equal(result["static_image_emb"][::16, 0], [0, 33, 66, 100])
    # First temporal sine component encodes the source time, not sample rank.
    np.testing.assert_allclose(result["static_pos_emb"][::16, 0], np.sin([0, 33, 66, 100]), atol=1e-7)


@pytest.mark.parametrize(
    "change",
    [
        {"budget": 31},
        {"budget": True},
        {"pool_type": "median"},
        {"num_views": 2},
        {"use_state_emb": True},
        {"use_pos_emb": "true"},
        {"memory_token_dim": 1024},
        {"representation_type": "recurrent"},
        {"integration_type": "context"},
        {"perceptual_memory": {"type": "token_dropping"}},
    ],
)
def test_invalid_config_rejected(change):
    config = load_memory_config()
    config.update(change)
    with pytest.raises(ValueError):
        validate_memory_config(config)


@pytest.mark.parametrize(
    "path,value",
    [
        (("memory_feature", "pos", "input_dim"), 7),
        (("memory_feature", "img", "input_dim"), 0),
        (("memory_feature", "pos", "hidden_dim"), 1.5),
        (("attention", "head_dim"), 255),
        (("attention", "num_heads"), True),
        (("attention", "rope_base"), float("nan")),
        (("attention", "output_init_std"), float("inf")),
    ],
)
def test_invalid_nested_fields(path, value):
    config = load_memory_config()
    section = config
    for name in path[:-1]:
        section = section[name]
    section[path[-1]] = value
    with pytest.raises(ValueError):
        validate_memory_config(config)


@pytest.mark.parametrize("text", ["", "- list\n- instead", "budget: 512\n", "attention: null\n"])
def test_malformed_yaml_has_clear_error(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="Memory config"):
        load_memory_config(path)


def test_typo_and_missing_nested_field_rejected():
    config = load_memory_config()
    bad = copy.deepcopy(config)
    bad["budegt"] = 512
    with pytest.raises(ValueError):
        validate_memory_config(bad)
    config["memory_feature"]["pos"].pop("hidden_dim")
    with pytest.raises(ValueError):
        validate_memory_config(config)


@pytest.mark.parametrize("shape,step", [((3, 16, 2), 3), ((0, 16, 2), 0), ((3, 8, 2), 0), ((3, 16, 1), 0)])
def test_missing_frame_or_wrong_feature_shape(shape, step):
    with pytest.raises(ValueError):
        prepare_frame_sampling(np.zeros(shape, np.float32), step, small_config())


def test_nonfinite_observed_features_rejected():
    features = np.zeros((3, 16, 2), np.float32)
    features[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare_frame_sampling(features, 2, small_config())
