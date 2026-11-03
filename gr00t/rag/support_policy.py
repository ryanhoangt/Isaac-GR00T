"""Stateless RPC adapter: each simulator owns its episode's gate state."""

from dataclasses import asdict
import time

import torch

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.rag.diagnosis_protocol import digest
from gr00t.rag.dual_mode_data import pack_context
from gr00t.rag.dual_mode_diagnosis import rng_devices, support_scores, synchronize
from gr00t.rag.dual_mode_memory import milestone_index
from gr00t.rag.support_gate import GateConfig, decide


class SupportPolicy:
    def __init__(self, policy, calibration, config=GateConfig()):
        self.policy, self.calibration, self.config = policy, calibration, config
        head = policy.vla.action_head
        if (
            not getattr(head.config, "dual_mode", False)
            or head.action_horizon != 16
            or head.num_inference_timesteps != 4
        ):
            raise ValueError("Support switching requires a dual-mode checkpoint, horizon 16, and four Euler steps")

    @torch.inference_mode()
    def predict(self, request):
        policy, model = self.policy, self.policy.vla
        strategy, variant = request.get("strategy", "adaptive"), request.get("variant", "q95")
        episode_id = request.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("An explicit episode_id is required for gate state isolation")
        if strategy not in ("action", "adaptive"):
            raise ValueError("strategy must be action or adaptive")
        seed = request.get("flow_seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("Invalid flow_seed")
        synchronize(model.device)
        started = time.perf_counter()
        batch = policy.collator([policy.transforms(request["observations"].copy())])
        task = batch.pop("tasks")[0]
        query = policy.retriever(model, batch)[0]
        entry = policy.memory[task]
        keys = entry["keys"].to(query)
        scores = query @ keys.T
        nearest = int(scores.argmax())
        support = support_scores(query, keys[nearest], scores[nearest])
        high, low = self.calibration.thresholds(task, variant)
        binding = digest(
            dict(
                task=task,
                episode_id=episode_id,
                strategy=strategy,
                variant=variant,
                gate=asdict(self.config),
                calibration=self.calibration.document["signature"],
            )
        )
        incoming = request.get("gate_state") or {}
        if incoming and (set(incoming) != {"binding", "state"} or incoming["binding"] != binding):
            raise ValueError("Gate state belongs to a different task/variant/calibration/configuration")
        future = milestone_index(entry, nearest, policy.lookahead)
        if strategy == "adaptive":
            state, event = decide(
                support["support_fp32"], high, low, incoming.get("state"), self.config, nearest, future
            )
        else:
            state = dict(mode="action", goal_chunks=0, cooldown=0, above=0, anchor_index=None, goal_index=None)
            event = "control_action"
        mode = state["mode"]
        anchor = state["anchor_index"] if mode == "goal" else nearest
        goal = state["goal_index"] if mode == "goal" else future
        if not (0 <= anchor < len(keys) and 0 <= goal < len(keys)):
            raise ValueError("Gate anchor is outside reference memory")
        if entry["episode_id"][anchor] != entry["episode_id"][goal]:
            raise ValueError("Gate milestone changed demonstration")
        if goal != milestone_index(entry, anchor, policy.lookahead):
            raise ValueError("Gate milestone is not the configured forward anchor")
        selected = goal if mode == "goal" else anchor
        context, actions = pack_context(
            [
                (
                    entry["backbone_features"][selected],
                    entry["backbone_attention_mask"][selected],
                    entry["action"][selected],
                )
            ],
            model.device,
            model.action_head.dtype,
        )
        batch.update(
            context_mode=torch.tensor([int(mode == "goal")]), ret_actions=actions, ret_backbone_output=context
        )
        with torch.random.fork_rng(devices=rng_devices(model.device)):
            torch.manual_seed(seed)
            normalized = GR00T_N1_5.get_action(model, batch).action_pred
        result = policy.transforms.unapply({"action": normalized[0].float().cpu()})
        synchronize(model.device)
        return dict(
            action=result,
            normalized_actions=normalized[0, :, :7].float().cpu().numpy(),
            task=task,
            mode=mode,
            flow_seed=seed,
            strategy=strategy,
            variant=variant,
            gate_state=dict(binding=binding, state=state),
            gate_event=event,
            threshold_enter=high,
            threshold_exit=low,
            nearest_index=nearest,
            nearest_episode=int(entry["episode_id"][nearest]),
            nearest_goal_index=future,
            nearest_frame=int(entry["frame_index"][nearest]),
            anchor_index=anchor,
            goal_index=goal,
            retrieved_episode=int(entry["episode_id"][anchor]),
            retrieved_frame=int(entry["frame_index"][anchor]),
            goal_frame=int(entry["frame_index"][goal]),
            query_ms=(time.perf_counter() - started) * 1000,
            **support,
        )
