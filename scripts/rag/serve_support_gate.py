"""Dual-mode support-gated server; the legacy diagnosis server is unchanged."""

from dataclasses import dataclass
import os
from pathlib import Path

import tyro

from gr00t.eval.robot import RobotInferenceServer
from gr00t.rag.support_calibration import Calibration
from gr00t.rag.support_gate import GateConfig
from gr00t.rag.support_policy import SupportPolicy
from scripts.rag.serve_dual_mode import Args as PolicyArgs, load_policy


@dataclass
class Args:
    checkpoint: Path
    calibration: Path
    memory: Path = Path("outputs/dual_mode/long_reference_diagnosis")
    retriever: Path = Path("ra-vla-official-assets/libero_gos")
    port: int = 8097
    server_id: str = "standalone"
    max_goal_chunks: int = 2
    cooldown_chunks: int = 1
    enter_consecutive: int = 1


def main(args):
    policy = load_policy(PolicyArgs(checkpoint=args.checkpoint, memory=args.memory, retriever=args.retriever))
    calibration = Calibration(args.calibration, policy.memory)
    config = GateConfig(args.max_goal_chunks, args.cooldown_chunks, args.enter_consecutive)
    support = SupportPolicy(policy, calibration, config)
    server = RobotInferenceServer(policy, "127.0.0.1", args.port)
    server.register_endpoint("get_action_support", support.predict)
    server.register_endpoint(
        "server_info",
        lambda: dict(
            server_id=args.server_id,
            pid=os.getpid(),
            checkpoint=str(args.checkpoint.resolve()),
            calibration_signature=calibration.document["signature"],
        ),
        requires_input=False,
    )
    server.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
