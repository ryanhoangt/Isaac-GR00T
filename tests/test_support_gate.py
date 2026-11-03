from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gr00t.rag.diagnosis_protocol import digest
from gr00t.rag.support_calibration import Calibration, build, calibrate_task, weighted_quantile
from gr00t.rag.support_gate import GateConfig, decide
from gr00t.rag.support_policy import SupportPolicy


def test_hysteresis_pins_anchor_caps_bursts_and_enforces_cooldown():
    config = GateConfig(max_goal_chunks=2, cooldown_chunks=2)
    state, event = decide(0.3, 0.2, 0.1, {}, config, 1, 5)
    assert (state["mode"], event) == ("goal", "enter_goal")
    state, event = decide(0.15, 0.2, 0.1, state, config, 8, 12)
    assert (state["anchor_index"], state["goal_index"], state["goal_chunks"]) == (1, 5, 2)
    state, event = decide(0.3, 0.2, 0.1, state, config, 8, 12)
    assert event == "exit_cap" and state["mode"] == "action"
    state, event = decide(0.3, 0.2, 0.1, state, config, 8, 12)
    assert event == "cooldown" and state["mode"] == "action"
    state, event = decide(0.3, 0.2, 0.1, state, config, 8, 12)
    assert event == "enter_goal" and state["anchor_index"] == 8
    state, event = decide(0.1, 0.2, 0.1, state, config, 9, 13)
    assert event == "exit_support" and state["mode"] == "action"


def test_threshold_boundaries_and_consecutive_enter():
    config = GateConfig(enter_consecutive=2)
    state, event = decide(0.2, 0.2, 0.1, {}, config, 0, 1)
    assert event == "supported_action"
    state, event = decide(0.21, 0.2, 0.1, state, config, 0, 1)
    assert event == "above_pending"
    state, event = decide(0.19, 0.2, 0.1, state, config, 0, 1)
    assert state["above"] == 0
    state, _ = decide(0.3, 0.2, 0.1, state, config, 0, 1)
    state, event = decide(0.3, 0.2, 0.1, state, config, 0, 1)
    assert event == "enter_goal"


@pytest.mark.parametrize(
    "score,high,low", [(float("nan"), 0.2, 0.1), (0.1, 0.1, 0.2), (-1, 0.2, 0.1), (0.1, 3, 0.1)]
)
def test_gate_rejects_invalid_scores(score, high, low):
    with pytest.raises(ValueError):
        decide(score, high, low, {}, GateConfig(), 0, 1)


def test_invalid_gate_state_and_config():
    with pytest.raises(ValueError):
        GateConfig(cooldown_chunks=0)
    with pytest.raises(ValueError):
        decide(0.3, 0.2, 0.1, dict(mode="goal"), GateConfig(), 0, 1)
    with pytest.raises(ValueError):
        decide(0.3, 0.2, 0.1, dict(anchor_index=1), GateConfig(), 0, 1)


def entry():
    torch.manual_seed(4)
    count = 15
    return dict(
        keys=F.normalize(torch.randn(count, 8), dim=-1).to(torch.bfloat16),
        episode_id=torch.arange(3).repeat_interleave(5),
        frame_index=torch.tensor([0, 4, 8, 12, 15] * 3),
        trajectory_length=torch.full((count,), 16),
        action=torch.randn(count, 16, 8),
        backbone_features=torch.randn(count, 2, 8),
        backbone_attention_mask=torch.ones(count, 2),
    )


def test_lodo_excludes_entire_query_demo_and_weights_demos_equally():
    data = entry()
    keep = ~((data["episode_id"] == 2) & (data["frame_index"] == 8))
    data = {k: v[keep] for k, v in data.items()}
    rows = calibrate_task(data, query_stride=8, device="cpu")
    assert len(rows) == 8
    assert all(r["query_episode"] != r["retrieved_episode"] for r in rows)
    for episode in range(3):
        assert sum(r["weight"] for r in rows if r["query_episode"] == episode) == pytest.approx(1 / 3)
    assert len({r["weight"] for r in rows}) == 2
    assert all(0 <= r["support_fp32"] <= 2 for r in rows)
    assert weighted_quantile([0, 0, 0, 1], [1 / 6, 1 / 6, 1 / 6, 1 / 2], 0.75) == 1
    with pytest.raises(ValueError):
        weighted_quantile([1], [0], 0.9)


def test_calibration_content_and_buffer_provenance(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    data = entry()
    torch.save(data, memory / "task.pt")
    (memory / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="completed",
                frame_stride=4,
                tasks={"task": dict(file="task.pt", episode_ids=[0, 1, 2], frames=15)},
            )
        )
    )
    document, rows = build(memory, query_stride=8, device="cpu")
    assert len(rows) == 9
    assert (
        document["tasks"]["task"]["exit"] <= document["tasks"]["task"]["q90"] <= document["tasks"]["task"]["q95"]
    )
    from gr00t.rag.dual_mode_memory import MilestoneMemory

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="CUDA"):
        Calibration(path, MilestoneMemory(memory))
    # Synthetic artifact for exercising fingerprint checks, without GPU hardware.
    document["settings"]["device_type"] = "cuda"
    document["signature"] = digest({k: v for k, v in document.items() if k != "signature"})
    path.write_text(json.dumps(document))
    calibration = Calibration(path, MilestoneMemory(memory))
    assert calibration.thresholds("Task!", "q95")[0] == document["tasks"]["task"]["q95"]
    changed = deepcopy(document)
    changed["tasks"]["task"]["q95"] += 0.1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="signature"):
        Calibration(path, MilestoneMemory(memory))
    path.write_text(json.dumps(document))
    data["keys"][0, 0] += 1
    torch.save(data, memory / "task.pt")
    with pytest.raises(ValueError, match="memory differs"):
        Calibration(path, MilestoneMemory(memory))


def fake_policy(monkeypatch):
    memory = dict(
        keys=torch.tensor([[0, 1]] * 9, dtype=torch.bfloat16),
        episode_id=torch.arange(3).repeat_interleave(3),
        frame_index=torch.tensor([0, 16, 32] * 3),
        trajectory_length=torch.full((9,), 33),
        action=torch.randn(9, 16, 8),
        backbone_features=torch.randn(9, 2, 8),
        backbone_attention_mask=torch.ones(9, 2),
    )
    calls = []

    def get_action(model, batch):
        mode = int(batch["context_mode"][0])
        calls.append(mode)
        return SimpleNamespace(action_pred=torch.randn(1, 16, 8) + mode)

    monkeypatch.setattr("gr00t.rag.support_policy.GR00T_N1_5.get_action", get_action)

    class Transforms:
        def __call__(self, observations):
            return observations

        def unapply(self, value):
            return {
                "action." + key: value["action"][:, i : i + 1].numpy()
                for i, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
            }

    head = SimpleNamespace(
        config=SimpleNamespace(dual_mode=True), action_horizon=16, num_inference_timesteps=4, dtype=torch.float32
    )
    model = SimpleNamespace(action_head=head, device=torch.device("cpu"))
    policy = SimpleNamespace(
        vla=model,
        memory={"task": memory},
        lookahead=16,
        transforms=Transforms(),
        collator=lambda samples: dict(tasks=["task"], **samples[0]),
        retriever=lambda model, batch: torch.tensor([[1, 0]], dtype=torch.bfloat16),
    )
    calibration = SimpleNamespace(
        document=dict(signature="test_calibration"), thresholds=lambda task, v: (0.2, 0.1)
    )
    return SupportPolicy(policy, calibration), calls


def test_single_head_pass_per_chunk_and_episode_isolation(monkeypatch):
    api, calls = fake_policy(monkeypatch)
    request = dict(observations={}, episode_id="episode1", variant="q95", flow_seed=17)
    first = api.predict(request)
    second = api.predict(dict(request, gate_state=first["gate_state"]))
    third = api.predict(dict(request, gate_state=second["gate_state"]))
    assert calls == [1, 1, 0]
    assert first["anchor_index"] == second["anchor_index"] and first["goal_index"] == second["goal_index"]
    assert third["gate_event"] == "exit_cap"
    np.testing.assert_array_equal(first["normalized_actions"], api.predict(request)["normalized_actions"])
    for field, value in (("episode_id", "episode2"), ("variant", "q90")):
        with pytest.raises(ValueError, match="different"):
            api.predict(dict(request, gate_state=first["gate_state"], **{field: value}))
    action = api.predict(dict(request, strategy="action"))
    assert action["mode"] == "action" and action["gate_event"] == "control_action"


def test_gate_replay_audit_uses_actual_executed_steps(monkeypatch, tmp_path):
    from scripts.rag.support_gate_report import audit_episode

    api, _ = fake_policy(monkeypatch)
    case = dict(name="adaptive_q95", kind="support", strategy="adaptive", variant="q95")
    signature = digest(dict(run="test_run", case=case, task=0, initial=0))
    state, queries = {}, []
    for step in (0, 16, 32, 48):
        result = api.predict(dict(observations={}, episode_id=signature, gate_state=state, flow_seed=step))
        result.pop("action")
        result["control_step"] = step
        queries.append(result)
        state = result["gate_state"]
    modes = np.zeros(51, dtype=np.int8)
    modes[:32] = 1
    modes[48:] = 1
    np.savez(tmp_path / "trace.npz", executed_actions=np.zeros((51, 7)), ee=np.zeros((52, 3)), goal_mode=modes)
    episode = dict(
        case=case,
        task="task",
        task_id=0,
        initial_index=0,
        signature=signature,
        directory=str(tmp_path),
        control_steps=51,
        goal_control_steps=35,
        queries=queries,
    )
    manifest = dict(
        settings=dict(gate=asdict(GateConfig())),
        calibration=dict(signature="test_calibration", tasks={"task": dict(exit=0.1, q95=0.2)}),
    )
    audit_episode(episode, manifest)
    episode["goal_control_steps"] = 48
    with pytest.raises(ValueError, match="accounting"):
        audit_episode(episode, manifest)
