"""Per-request mode/context control. Support scores are logged, never used to gate."""

import time

import numpy as np
import torch
import torch.nn.functional as F

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.rag.dual_mode_data import pack_context
from gr00t.rag.dual_mode_memory import milestone_index


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_devices(device):
    return [device.index] if device.type == "cuda" else []


def support_scores(query, key, legacy_similarity):
    # Re-normalize the stored BF16 vectors in FP32 before dotting. This does not
    # restore quantized embeddings, but avoids BF16 score accumulation/rounding.
    similarity = (F.normalize(query.float(), dim=-1) * F.normalize(key.float(), dim=-1)).sum().clamp(-1, 1)
    return {
        "similarity_bf16": float(legacy_similarity),
        "support_bf16": 1 - float(legacy_similarity),
        "similarity_fp32": float(similarity),
        "support_fp32": 1 - float(similarity),
    }


def action_matrix(actions):
    return np.concatenate(
        [actions[f"action.{key}"] for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")], axis=-1
    )


class DiagnosticPolicy:
    def __init__(self, policy):
        self.policy = policy
        head = policy.vla.action_head
        if head.action_horizon != 16 or head.num_inference_timesteps != 4:
            raise ValueError("Diagnosis protocol requires horizon 16 and four Euler steps")

    @torch.inference_mode()
    def predict(self, request):
        policy, model = self.policy, self.policy.vla
        mode = request.get("mode", "action")
        if mode not in ("action", "goal", "copy"):
            raise ValueError("mode must be action, goal, or copy")
        if mode == "goal" and not getattr(model.action_head.config, "dual_mode", False):
            raise ValueError("Goal inference requires a dual-mode checkpoint")
        seed = request.get("flow_seed", 0)
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("flow_seed must be an integer in [0, 2**31)")
        synchronize(model.device)
        started = time.perf_counter()
        batch = policy.collator([policy.transforms(request["observations"].copy())])
        task = batch.pop("tasks")[0]
        query = policy.retriever(model, batch)[0]
        entry = policy.memory[task]
        retrieval_started = time.perf_counter()
        keys = entry["keys"].to(query)
        scores = query @ keys.T
        nearest = int(scores.argmax())
        anchor = request.get("anchor_index", nearest)
        if not isinstance(anchor, int) or not 0 <= anchor < len(keys):
            raise ValueError("Invalid anchor_index")
        goal = request.get("goal_index", milestone_index(entry, anchor, policy.lookahead))
        if not isinstance(goal, int) or not 0 <= goal < len(keys):
            raise ValueError("Invalid goal_index")
        if entry["episode_id"][goal] != entry["episode_id"][anchor]:
            raise ValueError("Fixed milestone must belong to the anchor demonstration")
        support = support_scores(query, keys[nearest], scores[nearest])
        synchronize(model.device)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        selected = goal if mode == "goal" else anchor
        row = (
            entry["backbone_features"][selected],
            entry["backbone_attention_mask"][selected],
            entry["action"][selected],
        )
        if mode == "copy":
            normalized = row[2].to(model.device, dtype=model.action_head.dtype)[None]
        else:
            context, actions = pack_context([row], model.device, model.action_head.dtype)
            batch.update(
                context_mode=torch.tensor([int(mode == "goal")]), ret_actions=actions, ret_backbone_output=context
            )
            with torch.random.fork_rng(devices=rng_devices(model.device)):
                torch.manual_seed(seed)
                normalized = GR00T_N1_5.get_action(model, batch).action_pred
        result = policy.transforms.unapply({"action": normalized[0].float().cpu()})
        synchronize(model.device)
        return {
            "action": result,
            "normalized_actions": normalized[0, :, :7].float().cpu().numpy(),
            "task": task,
            "mode": mode,
            "flow_seed": seed,
            "nearest_index": nearest,
            "anchor_index": anchor,
            "goal_index": goal,
            "nearest_episode": int(entry["episode_id"][nearest]),
            "nearest_frame": int(entry["frame_index"][nearest]),
            "retrieved_episode": int(entry["episode_id"][anchor]),
            "retrieved_frame": int(entry["frame_index"][anchor]),
            "goal_frame": int(entry["frame_index"][goal]),
            "retrieval_ms": retrieval_ms,
            "query_ms": (time.perf_counter() - started) * 1000,
            **support,
        }

    @torch.inference_mode()
    def probe_cached(self, task, current_index, goal_index, seed):
        """Reference-state probe, with no VLM rerun and a fixed Gaussian source."""
        policy, head = self.policy, self.policy.vla.action_head
        entry = policy.memory[task]
        if not all(key in entry for key in ("state", "embodiment_id")):
            raise ValueError("Probes require a memory built with --diagnosis")
        visual = {
            "backbone_features": entry["backbone_features"][current_index : current_index + 1].to(
                head.device, dtype=head.dtype
            ),
            "backbone_attention_mask": entry["backbone_attention_mask"][current_index : current_index + 1].to(
                head.device
            ),
        }
        context, actions = pack_context(
            [
                (
                    entry["backbone_features"][goal_index],
                    entry["backbone_attention_mask"][goal_index],
                    entry["action"][goal_index],
                )
            ],
            head.device,
            head.dtype,
        )
        from transformers.feature_extraction_utils import BatchFeature

        inputs = BatchFeature(
            data={
                "state": entry["state"][current_index : current_index + 1].to(head.device, dtype=head.dtype),
                "embodiment_id": entry["embodiment_id"][current_index : current_index + 1].to(head.device),
                "context_mode": torch.ones(1, device=head.device),
                "ret_actions": actions,
                "ret_backbone_output": context,
            }
        )
        with torch.random.fork_rng(devices=rng_devices(head.device)):
            torch.manual_seed(seed)
            output = head.get_action(BatchFeature(data=visual), inputs).action_pred
        denormalized = policy.transforms.unapply({"action": output[0].float().cpu()})
        return action_matrix(denormalized)
