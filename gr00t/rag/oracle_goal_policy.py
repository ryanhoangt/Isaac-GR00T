"""Isolated goal head with live same-scene milestone images; no uncertainty gate."""

from collections import OrderedDict
import time

import torch

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.rag.dual_mode_data import encode_context, pack_context
from gr00t.rag.dual_mode_diagnosis import rng_devices, support_scores, synchronize
from gr00t.rag.oracle_goal_protocol import array_hash


class OracleGoalPolicy:
    def __init__(self, model, retriever, memory, transforms, collator, modalities):
        self.model, self.retriever, self.memory = model, retriever, memory
        self.transforms, self.collator, self.modalities = transforms, collator, modalities
        self.cache = OrderedDict()
        if not hasattr(model.action_head, "goal_branch"):
            raise ValueError("This diagnostic requires an isolated goal branch")
        if model.action_head.action_horizon != 16 or model.action_head.num_inference_timesteps != 4:
            raise ValueError("Expected horizon 16 and four Euler steps")

    def get_modality_config(self):
        return self.modalities

    def _batch(self, observations):
        batch = self.collator([self.transforms(observations.copy())])
        return batch.pop("tasks")[0], batch

    def get_action(self, observations):
        return self.predict(dict(observations=observations, mode="action", flow_seed=0))["action"]

    @torch.inference_mode()
    def predict(self, request):
        mode = request["mode"]
        if mode not in ("action", "goal"):
            raise ValueError("mode must be action or goal")
        seed = request["flow_seed"]
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("Invalid flow_seed")
        synchronize(self.model.device)
        started = time.perf_counter()
        task, batch = self._batch(request["observations"])
        details = {}
        if mode == "action":
            query = self.retriever(self.model, batch)[0]
            entry = self.memory[task]
            keys = entry["keys"].to(query)
            scores = query @ keys.T
            index = int(scores.argmax())
            row = (
                entry["backbone_features"][index],
                entry["backbone_attention_mask"][index],
                entry["action"][index],
            )
            context, actions = pack_context([row], self.model.device, self.model.action_head.dtype)
            details.update(memory_index=index, **support_scores(query, keys[index], scores[index]))
        else:
            observations = request["goal_observations"]
            goal_task, goal_batch = self._batch(observations)
            if goal_task != task:
                raise ValueError("Goal/current language task mismatch")
            # Image content participates in the key: no stale goal cache if a
            # caller reuses a provenance ID for a different frame.
            key = (task, array_hash(observations["video.image"]), array_hash(observations["video.wrist_image"]))
            context = self.cache.pop(key, None)
            if context is None:
                context = encode_context(self.model, goal_batch)
            self.cache[key] = context
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
            # Neither reference actions nor reference proprioception enters the
            # head. Goal VLM tokens use the same two views/language as training.
            actions = torch.zeros(
                1,
                16,
                self.model.action_head.action_dim,
                device=self.model.device,
                dtype=self.model.action_head.dtype,
            )
            details.update(goal_image_sha256=key[1], goal_wrist_sha256=key[2])
        batch.update(
            context_mode=torch.tensor([int(mode == "goal")]), ret_actions=actions, ret_backbone_output=context
        )
        with torch.random.fork_rng(devices=rng_devices(self.model.device)):
            torch.manual_seed(seed)
            normalized = GR00T_N1_5.get_action(self.model, batch).action_pred
        action = self.transforms.unapply({"action": normalized[0].float().cpu()})
        synchronize(self.model.device)
        return dict(
            action=action,
            mode=mode,
            task=task,
            flow_seed=seed,
            normalized_actions=normalized[0, :, :7].float().cpu().numpy(),
            query_ms=(time.perf_counter() - started) * 1000,
            **details,
        )
