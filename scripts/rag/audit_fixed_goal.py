"""Check replay/goal identity and video completeness using robot observations only."""

import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np

from gr00t.rag.fixed_goal_protocol import load_reference, observation_identity, reference_observation
from gr00t.rag.oracle_goal_protocol import array_hash


def main(output):
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Use a Slurm CPU/GPU worker")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Wait for the run to complete")
    paths = sorted(output.glob("*/*/episode.json"))
    if len(paths) != manifest["expected_cases"]:
        raise ValueError("Incomplete cases")
    video_frames, goal_queries, groups = 0, 0, {}
    for path in paths:
        row = json.loads(path.read_text())
        assert row["complete"] and row["signature"] == manifest["signature"]
        assert row["control_steps"] == row["horizon"] == 16
        assert row["fixed_goal_frame"] == row["prefix"] + 16
        checks = json.loads((path.parent / "replay_checks.json").read_text())
        assert len(checks) == row["prefix"] + 1 and all(c["matched"] for c in checks)
        reference = load_reference(Path(row["reference"]) / "trace.npz", row["prefix"], row["horizon"])
        goal = reference_observation(reference, row["fixed_goal_frame"])
        assert row["goal_observation_identity"] == observation_identity(goal)
        assert row["start_observation_identity"] == observation_identity(
            reference_observation(reference, row["prefix"])
        )
        with np.load(path.parent / "trace.npz", allow_pickle=False) as data:
            allowed = {
                "agent",
                "wrist",
                "ee",
                "quat",
                "gripper",
                "actions",
                "reference_actions",
                "goal_agent",
                "goal_wrist",
                "goal_ee",
                "goal_quat",
                "goal_gripper",
            }
            assert set(data.files) == allowed
            assert len(data["agent"]) == 17 and len(data["actions"]) == 16
            assert np.array_equal(data["goal_agent"], goal["agentview_image"])
            assert np.array_equal(data["goal_wrist"], goal["robot0_eye_in_hand_image"])
        for query in row["queries"]:
            assert query["fixed_goal_frame"] == row["fixed_goal_frame"]
            if row["mode"] == "goal":
                assert query["goal_image_sha256"] == array_hash(goal["agentview_image"][::-1, ::-1][None])
                assert query["goal_wrist_sha256"] == array_hash(goal["robot0_eye_in_hand_image"][::-1, ::-1][None])
                goal_queries += 1
        with imageio.get_reader(str(path.parent / "rollout.mp4")) as reader:
            count = reader.count_frames()
        assert count == 17
        video_frames += count
        groups.setdefault((row["suite"], row["initial_index"]), {})[row["mode"]] = row
    for pair in groups.values():
        assert pair.keys() == {"action", "goal"}
        assert pair["action"]["start_observation_identity"] == pair["goal"]["start_observation_identity"]
    result = dict(
        status="passed",
        cases=len(paths),
        pairs=len(groups),
        goal_queries_audited=goal_queries,
        videos=len(paths),
        total_video_frames=video_frames,
        checks=[
            "every prefix frame matched",
            "mode starts identical",
            "goal fixed to reference t+16",
            "exact goal images reached the policy",
            "output traces contain robot observations only",
            "17 frames per video",
        ],
    )
    (output / "observation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
