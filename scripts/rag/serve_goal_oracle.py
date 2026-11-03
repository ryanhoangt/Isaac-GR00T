"""GPU-worker server for the isolated goal-controller rollout diagnostic."""

import argparse
import os
from pathlib import Path

import torch

from examples.Libero.custom_data_config import LiberoDataConfig
from gr00t.eval.robot import RobotInferenceServer
from gr00t.rag.goal_control import GoalControlRAVLA, assert_warm_owner, owner_digest
from gr00t.rag.oracle_goal_policy import OracleGoalPolicy
from gr00t.rag.ravla import RAVLADataCollator
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything
from scripts.rag.train_dual_mode import check_worker


def main(args):
    check_worker()
    seed_everything(0)
    model = GoalControlRAVLA.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        tune_llm=False,
        tune_visual=False,
    ).eval()
    count = assert_warm_owner(model.action_head, args.baseline)
    config = LiberoDataConfig()
    transforms = config.transform()
    metadata = torch.load(Path(args.checkpoint) / "metadata.pt", map_location="cpu", weights_only=False)
    transforms.set_metadata(metadata["new_embodiment"])
    transforms.eval()
    retriever = RetrieverWrapperWithHead.from_pretrained(
        args.retriever,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    memory = torch.load(Path(args.retriever) / "memory_gos.pt", map_location="cpu", mmap=True, weights_only=True)
    policy = OracleGoalPolicy(model, retriever, memory, transforms, RAVLADataCollator(), config.modality_config())
    info = dict(
        server_id=args.server_id,
        checkpoint=str(Path(args.checkpoint).resolve()),
        pid=os.getpid(),
        owner_sha256=owner_digest(model.action_head),
        verified_owner_tensors=count,
    )
    server = RobotInferenceServer(policy, "127.0.0.1", args.port)
    server.register_endpoint("predict_oracle", policy.predict)
    server.register_endpoint("server_info", lambda: info, requires_input=False)
    print("ORACLE_SERVER", info, flush=True)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--retriever", default="ra-vla-official-assets/libero_gos")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--server-id", required=True)
    main(parser.parse_args())
