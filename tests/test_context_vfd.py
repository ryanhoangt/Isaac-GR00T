"""Scientific invariants for context-conditioned VFD; tiny real RA-VLA heads."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers.feature_extraction_utils import BatchFeature

from gr00t.rag import context_vfd as ctx
from scripts.rag.score_context import load_observation, write_result


def test_shared_actor_trajectory_and_exact_quadrature():
    initial = torch.full((2, 3, 9), 2.0)
    scorer_states = []

    def actor(x, time):
        return torch.ones_like(x)

    def scorer(x, time):
        scorer_states.append((time, x.clone()))
        return torch.zeros_like(x)

    terminal, scores, trace = ctx.integrate_context_vfd(actor, scorer, initial, 4)
    torch.testing.assert_close(terminal, initial + 1)
    torch.testing.assert_close(scores, torch.full((2,), 21 * (1 / 12 + 1 / 4 + 3 / 4)))
    assert trace["flow_times"] == [0.25, 0.5, 0.75]
    torch.testing.assert_close(trace["weighted_disagreement"].sum(1), scores)
    for time, state in scorer_states:
        torch.testing.assert_close(state, initial + time)
    assert ctx.flow_grid(8, [0.5, 0.75]) == {4: 0.25, 6: 0.75}


def test_padding_excluded_but_integrated_and_fp32_accumulation():
    initial = torch.zeros(1, 16, 32, dtype=torch.bfloat16)

    def actor(x, time):
        result = torch.zeros_like(x)
        result[..., 7:] = 100
        return result

    terminal, scores, _ = ctx.integrate_context_vfd(
        actor, lambda x, t: torch.zeros_like(x), initial, 4
    )
    assert scores.dtype == torch.float32 and scores.item() == 0
    assert terminal.dtype == torch.bfloat16 and terminal[..., 7:].eq(100).all()


@pytest.mark.parametrize("times", [[1], [-0.1], [0.2], [], [0], [float("nan")]])
def test_invalid_flow_times(times):
    with pytest.raises(ValueError):
        ctx.flow_grid(4, times)


def test_reject_nonfinite_and_bad_shapes():
    initial = torch.zeros(1, 2, 8)
    with pytest.raises(FloatingPointError):
        ctx.integrate_context_vfd(
            lambda x, t: x * float("nan"), lambda x, t: x, initial, 4
        )
    with pytest.raises(ValueError):
        ctx.integrate_context_vfd(lambda x, t: x[..., :7], lambda x, t: x, initial, 4)


@pytest.fixture
def memory():
    torch.manual_seed(13)
    return {
        "task": {
            "keys": torch.eye(3),
            "action": torch.randn(3, 2, 9),
            "backbone_features": torch.randn(3, 4, 8),
            "backbone_attention_mask": torch.ones(3, 4, dtype=torch.bool),
            "pixel_values": torch.zeros(3, 2, 3, 2, 2),
        }
    }


def test_retrieval_matches_original_and_honors_fixed_context_and_mask(memory):
    from gr00t.rag.retriever import retrieve_from_memory

    query = torch.tensor([[0.1, 0.9, 0.5]])
    context = ctx.select_context(memory, "task", query, 2, "cpu", torch.float32)
    (_, _, actions), (features, mask) = retrieve_from_memory(memory, 2, ["task"], query)
    assert context.indices == (1, 2)
    torch.testing.assert_close(context.actions, actions, rtol=0, atol=0)
    torch.testing.assert_close(
        context.backbone_output.backbone_features, features, rtol=0, atol=0
    )
    torch.testing.assert_close(context.backbone_output.backbone_attention_mask, mask)
    masked = ctx.select_context(
        memory,
        "task",
        query,
        1,
        "cpu",
        torch.float32,
        candidate_mask=[True, False, True],
    )
    assert masked.indices == (2,)
    fixed = ctx.select_context(
        memory, "task", query, 1, "cpu", torch.float32, indices=[0]
    )
    assert fixed.indices == (0,) and fixed.source == "explicit_indices"
    for kwargs in [
        dict(indices=[3]),
        dict(indices=[0, 0]),
        dict(candidate_mask=[False] * 3),
        dict(candidate_mask=[1, 0, 1]),
        dict(candidate_mask=[True]),
        dict(indices=[1], candidate_mask=[True, False, True]),
    ]:
        with pytest.raises(ValueError):
            ctx.select_context(memory, "task", query, 1, "cpu", torch.float32, **kwargs)


@pytest.fixture
def head():
    from gr00t.rag.ravla import RAVLAActionHead
    from gr00t.model.action_head.flow_matching_action_head import (
        FlowmatchingActionHeadConfig,
    )

    torch.manual_seed(17)
    config = FlowmatchingActionHeadConfig(
        action_dim=9,
        action_horizon=2,
        max_state_dim=3,
        max_num_embodiments=1,
        hidden_size=8,
        input_embedding_dim=8,
        backbone_embedding_dim=8,
        max_seq_len=2,
        num_target_vision_tokens=1,
        num_inference_timesteps=4,
        num_timestep_buckets=1000,
        add_pos_embed=True,
        use_vlln=True,
        diffusion_model_cfg=dict(
            num_attention_heads=2,
            attention_head_dim=4,
            output_dim=8,
            num_layers=2,
            dropout=0.2,
            interleave_self_attention=True,
            cross_attention_dim=8,
            positional_embeddings=None,
        ),
        vl_self_attention_cfg=dict(
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            dropout=0.2,
            positional_embeddings=None,
        ),
    )
    return RAVLAActionHead(config).eval()


@pytest.mark.parametrize("k", [1, 2])
def test_real_head_parity_and_no_mutation_of_raw_features(head, memory, k):
    context = ctx.select_context(
        memory, "task", torch.tensor([[0.1, 0.9, 0.5]]), k, "cpu", torch.float32
    )
    raw = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 4, 8),
            "backbone_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }
    )
    before = raw.backbone_features.clone()
    ret_before = context.backbone_output.backbone_features.clone()
    inputs = BatchFeature(
        data={
            "state": torch.randn(1, 1, 3),
            "embodiment_id": torch.tensor([0]),
            "ret_actions": context.actions,
            "ret_backbone_output": copy.deepcopy(context.backbone_output),
        }
    )
    with torch.inference_mode():
        expected = head.get_action(copy.deepcopy(raw), inputs).action_pred
        fn = ctx.velocity_function(
            head, raw, inputs.state, inputs.embodiment_id, context
        )
        initial = context.actions.reshape(1, k, 2, 9).mean(1)
        actual, scores, _ = ctx.integrate_context_vfd(fn, fn, initial, 4)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert scores.item() == 0
    torch.testing.assert_close(raw.backbone_features, before, rtol=0, atol=0)
    torch.testing.assert_close(
        context.backbone_output.backbone_features, ret_before, rtol=0, atol=0
    )


def test_second_head_uses_own_projectors(head, memory):
    other = copy.deepcopy(head)
    with torch.no_grad():
        other.action_decoder.layer2.b.add_(0.5)
        other.vlln.bias.add_(0.3)
    context = ctx.select_context(
        memory, "task", torch.tensor([[0.1, 0.9, 0.5]]), 1, "cpu", torch.float32
    )
    raw = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 4, 8),
            "backbone_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }
    )
    state, ids = torch.ones(1, 1, 3), torch.tensor([0])
    with torch.inference_mode():
        actor = ctx.velocity_function(head, raw, state, ids, context)
        scorer = ctx.velocity_function(other, raw, state, ids, context)
        _, score, _ = ctx.integrate_context_vfd(actor, scorer, context.actions, 4)
    assert score.item() > 0


def test_policy_shares_retrieval_and_backbone_and_preserves_rng(head, memory):
    calls = dict(backbone=0, retriever=0, transforms=0)
    raw = BatchFeature(
        data={
            "backbone_features": torch.ones(1, 4, 8),
            "backbone_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }
    )

    def backbone(inputs):
        calls["backbone"] += 1
        return raw

    def retriever(actor, inputs):
        calls["retriever"] += 1
        return torch.tensor([[0.1, 0.9, 0.5]])

    class Transforms:
        def __call__(self, data):
            calls["transforms"] += 1
            return data

        def unapply(self, data):
            return {"action.x": data["action"].numpy()}

    actor = SimpleNamespace(
        training=False,
        action_head=head,
        backbone=backbone,
        retriever=[retriever],
        memory=memory,
        k=1,
        prepare_input=lambda x: (
            None,
            BatchFeature(
                data={"state": torch.ones(1, 1, 3), "embodiment_id": torch.tensor([0])}
            ),
        ),
    )
    policy = SimpleNamespace(
        vla=actor, transforms=Transforms(), collator=lambda x: {"tasks": ["task"]}
    )
    scorer = copy.deepcopy(head)
    with torch.no_grad():
        scorer.action_decoder.layer2.b.add_(0.5)
    monitor = ctx.ContextVFDPolicy(policy, scorer, {"test": "pair"})
    rng = torch.get_rng_state().clone()
    result = monitor.get_action_with_uncertainty({})
    assert calls == dict(backbone=1, retriever=1, transforms=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert result["u_ctx"] > 0 and result["context"]["indices"] == [1]
    assert result["score_id"] == monitor.get_score_signature()["score_id"]
    repeat = monitor.get_action_with_uncertainty({})
    assert result["u_ctx"] == repeat["u_ctx"]
    np.testing.assert_array_equal(
        result["action"]["action.x"], repeat["action"]["action.x"]
    )
    signature = monitor.get_score_signature()
    signature["action_dimensions"].clear()
    assert len(monitor.get_score_signature()["action_dimensions"]) == 7
    torch.testing.assert_close(
        torch.as_tensor(result["u_ctx_per_dimension"]).sum(),
        torch.tensor(result["u_ctx"]),
        rtol=1e-5,
        atol=1e-6,
    )
    calls.update(backbone=0, retriever=0, transforms=0)
    numpy_rng = np.random.get_state()[1].copy()
    diagnosed = monitor.get_action_with_context_diagnostics({})
    assert calls == dict(backbone=1, retriever=1, transforms=1)
    assert diagnosed["u_ctx"] == result["u_ctx"]
    np.testing.assert_array_equal(
        diagnosed["action"]["action.x"], result["action"]["action.x"]
    )
    assert set(diagnosed["context_diagnostics"]) == {
        "random_same_task",
        "low_similarity_same_task",
    }
    for value in diagnosed["context_diagnostics"].values():
        assert value["context"]["indices"] != [1]
        assert value["normalized_action_distance_to_executed"] >= 0
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_rng, np.random.get_state()[1])
    repeated = monitor.get_action_with_context_diagnostics({})
    for key in diagnosed["context_diagnostics"]:
        assert (
            diagnosed["context_diagnostics"][key]["context"]
            == repeated["context_diagnostics"][key]["context"]
        )
        assert (
            diagnosed["context_diagnostics"][key]["u_ctx"]
            == repeated["context_diagnostics"][key]["u_ctx"]
        )


def make_checkpoint(path, head_weight=1.0, backbone_weight=1.0, shard=False):
    path.mkdir()
    cfg = dict(
        backbone_cfg={"x": 1}, action_head_cfg={"a": 2}, action_horizon=2, action_dim=9
    )
    (path / "config.json").write_text(json.dumps(cfg))
    tensors = {
        "backbone.weight": torch.full((2, 3), backbone_weight),
        "action_head.weight": torch.full((2, 3), head_weight),
    }
    name = "model-00001-of-00001.safetensors" if shard else "model.safetensors"
    save_file(tensors, path / name)
    if shard:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: name for k in tensors}})
        )
    return path


def test_checkpoint_audit_checks_content_metadata_and_independent_heads(
    tmp_path, monkeypatch
):
    actor = make_checkpoint(tmp_path / "actor")
    scorer = make_checkpoint(tmp_path / "scorer", head_weight=2.0, shard=True)
    retriever = tmp_path / "retriever"
    retriever.mkdir()
    (retriever / "config.json").write_text("{}")
    monkeypatch.setattr(ctx, "_metadata", lambda p: (None, {"norm": 1}))
    signature = ctx.validate_checkpoint_pair(actor, scorer, retriever)
    assert signature["actor_head_sha256"] != signature["scorer_head_sha256"]
    assert ctx._tensor_digest(actor, "backbone.") == ctx._tensor_digest(
        scorer, "backbone."
    )
    with pytest.raises(ValueError, match="different checkpoints"):
        ctx.validate_checkpoint_pair(actor, actor, retriever)
    same = make_checkpoint(tmp_path / "same")
    with pytest.raises(ValueError, match="identical"):
        ctx.validate_checkpoint_pair(actor, same, retriever)
    wrong = make_checkpoint(tmp_path / "wrong", head_weight=2.0, backbone_weight=3.0)
    with pytest.raises(ValueError, match="backbones differ"):
        ctx.validate_checkpoint_pair(actor, wrong, retriever)
    monkeypatch.setattr(
        ctx, "_metadata", lambda p: (None, {"norm": 2 if Path(p) == scorer else 1})
    )
    with pytest.raises(ValueError, match="normalization"):
        ctx.validate_checkpoint_pair(actor, scorer, retriever)


def test_load_scorer_head_is_strict_and_does_not_alias_or_touch_rng(tmp_path):
    head = torch.nn.Linear(3, 2).eval()
    path = tmp_path / "scorer"
    path.mkdir()
    save_file(
        {
            "action_head.weight": torch.ones(2, 3),
            "action_head.bias": torch.ones(2),
            "backbone.unused": torch.zeros(1),
        },
        path / "model.safetensors",
    )
    original = head.weight.clone()
    rng = torch.get_rng_state().clone()
    scorer = ctx.load_scorer_head(head, path)
    assert (
        scorer.weight.eq(1).all()
        and not scorer.training
        and not scorer.weight.requires_grad
    )
    torch.testing.assert_close(head.weight, original, rtol=0, atol=0)
    assert scorer.weight.data_ptr() != head.weight.data_ptr() and torch.equal(
        rng, torch.get_rng_state()
    )
    save_file({"action_head.weight": torch.ones(2, 3)}, path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        ctx.load_scorer_head(head, path)


def test_raw_observation_npz_and_atomic_output(tmp_path):
    obs = {
        f"state.{key}": np.zeros((1, 2 if key == "gripper" else 1))
        for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    }
    obs.update(
        {
            "video.image": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "video.wrist_image": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "annotation.human.action.task_description": np.asarray(["task"]),
        }
    )
    path = tmp_path / "observation.npz"
    np.savez(path, **obs)
    assert load_observation(path)["annotation.human.action.task_description"] == [
        "task"
    ]
    obs["state.x"] = np.zeros((2, 1))
    np.savez(path, **obs)
    with pytest.raises(ValueError, match="state.x"):
        load_observation(path)
    output = tmp_path / "result.json"
    write_result(output, {"score": np.asarray([1.0])})
    with pytest.raises(FileExistsError):
        write_result(output, {"score": 2})
    assert json.loads(output.read_text()) == {"score": [1.0]}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_result(invalid, {"score": float("nan")})
    assert not invalid.exists() and not list(tmp_path.glob(".ctx_*"))


@pytest.mark.parametrize("enabled", [False, True])
def test_server_preserves_normal_endpoint_and_optionally_adds_atomic_score_rpc(
    monkeypatch, enabled
):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts/rag/4_serve_ravla.py"
    spec = importlib.util.spec_from_file_location("serve_context_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    transforms = SimpleNamespace(set_metadata=lambda x: None, eval=lambda: None)
    config = SimpleNamespace(modality_config=lambda: {}, transform=lambda: transforms)
    monkeypatch.setattr(module, "load_data_config", lambda name: config)
    monkeypatch.setattr(module.torch, "load", lambda *a, **k: {"new_embodiment": {}})
    vla = SimpleNamespace(
        action_head=SimpleNamespace(),
        config=SimpleNamespace(),
        set_retrieval_components=lambda *a: None,
    )
    vla.eval = lambda: vla
    retriever = SimpleNamespace()
    retriever.eval = lambda: retriever
    monkeypatch.setattr(
        module, "RAVLA", SimpleNamespace(from_pretrained=lambda *a, **k: vla)
    )
    monkeypatch.setattr(
        module,
        "RetrieverWrapperWithHead",
        SimpleNamespace(from_pretrained=lambda *a, **k: retriever),
    )
    policy = SimpleNamespace(get_action=lambda obs: {"original": True})
    monkeypatch.setattr(module, "VLAPolicy", lambda **kwargs: policy)
    calls = []
    monitor = SimpleNamespace(
        get_action_with_uncertainty=lambda obs: {"u_ctx": 2.0},
        get_action_with_context_diagnostics=lambda obs: {"u_ctx": 2.0},
        get_score_signature=lambda: {"score_id": "test"},
    )

    def create_monitor(*args, **kwargs):
        calls.append((args, kwargs))
        return monitor

    monkeypatch.setattr(ctx.ContextVFDPolicy, "from_actor_policy", create_monitor)
    endpoints = {}

    class Server:
        def __init__(self, model, host, port):
            endpoints["get_action"] = model.get_action

        def register_endpoint(self, name, fn, **kwargs):
            endpoints[name] = fn

        def run(self):
            pass

    monkeypatch.setattr(module, "RobotInferenceServer", Server)
    module.main(
        module.ArgsConfig(
            checkpoint_path="actor", scorer_path="scorer" if enabled else None
        )
    )
    assert endpoints["get_action"]({}) == {"original": True}
    assert ("get_action_with_uncertainty" in endpoints) == enabled
    assert ("get_score_signature" in endpoints) == enabled
    assert len(calls) == int(enabled)
    if enabled:
        assert calls[0][0][0] is policy
        assert endpoints["get_action_with_uncertainty"]({}) == {"u_ctx": 2.0}
