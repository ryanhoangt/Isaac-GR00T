import json
import pprint
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
import tyro

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    # convert RGB -> BGR for OpenCV
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # show in separate windows
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


@dataclass
class GenerateConfig:
    task_suite_name: str = "libero_spatial"
    """LIBERO suite: libero_spatial, libero_object, libero_goal, libero_10, or libero_90."""
    num_steps_wait: int = 10
    """No-op steps for objects to stabilize before querying the policy."""
    num_trials_per_task: int = 5
    """Number of official initial states to evaluate per task."""
    port: int = 5555
    """Local inference server port."""
    headless: bool = False
    """Disable GUI observation previews."""
    output_dir: str = "./outputs/libero_eval"
    """Directory for logs, summary.json, and rollout videos."""
    seed: int = 0
    """Random seed for the simulator and evaluator."""
    save_videos: bool = True
    """Save camera replays as MP4 files."""
    request_timeout_ms: int = 60000
    """Maximum time to wait for an inference response."""


class GR00TPolicy:
    """GR00T Policy wrapper for Libero environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(self, host="localhost", port=5555, headless=False, timeout_ms=60000):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port, timeout_ms=timeout_ms)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless

    def get_action(self, observation_dict, lang: str):
        """Get action from GR00T policy given observation and language instruction."""
        obs_dict = self._process_observation(observation_dict, lang)
        # summarize_obs(obs_dict)
        action_chunk = self.policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, 0)

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(
        self, action_chunk: dict[str, np.array], idx: int = 0
    ) -> np.ndarray:
        """Convert GR00T action chunk to Libero format.

        Args:
            action_chunk: Dictionary of action components from GR00T policy
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


def eval_libero(cfg: GenerateConfig) -> None:
    from libero.libero import benchmark

    limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 600,
              "libero_10": 1000, "libero_90": 400}
    if cfg.task_suite_name not in limits:
        raise ValueError(f"Unsupported suite: {cfg.task_suite_name}")
    if cfg.num_trials_per_task < 1 or cfg.num_steps_wait < 0 or cfg.request_timeout_ms < 1:
        raise ValueError("Trials/timeout must be positive; wait steps must be nonnegative")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    # Stabilization steps count toward robosuite's internal termination horizon.
    # LIBERO replaces returned `done` with task success, hiding horizon termination.
    simulator_horizon = limits[cfg.task_suite_name] + cfg.num_steps_wait
    summary = {"status": "running", "config": asdict(cfg), "tasks": [],
               "max_action_steps": limits[cfg.task_suite_name], "simulator_horizon": simulator_horizon,
               "total_episodes": 0, "total_successes": 0, "success_rate": 0.0}

    def save_summary():
        count = summary["total_episodes"]
        summary["success_rate"] = summary["total_successes"] / count if count else 0.0
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)

    save_summary()
    try:
        with (output / f"libero_eval_{cfg.task_suite_name}.log").open("w") as log_file:
            for task_id in tqdm.tqdm(range(suite.n_tasks)):
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                if cfg.num_trials_per_task > len(states):
                    raise ValueError(f"Task {task_id} has only {len(states)} initial states")
                env, description = get_libero_env(task, resolution=256, horizon=simulator_horizon)
                policy = None
                result = {"task_id": task_id, "description": description, "episodes": [],
                          "successes": 0, "success_rate": 0.0}
                summary["tasks"].append(result)
                try:
                    env.seed(cfg.seed)
                    policy = GR00TPolicy(port=cfg.port, headless=cfg.headless,
                                         timeout_ms=cfg.request_timeout_ms)
                    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                        env.reset()
                        obs = env.set_init_state(states[episode_idx])
                        success = False
                        top_view, wrist_view = [], []
                        action_steps = 0
                        for t in range(limits[cfg.task_suite_name] + cfg.num_steps_wait):
                            if t < cfg.num_steps_wait:
                                obs, _, _, _ = env.step(get_libero_dummy_action())
                                continue
                            if cfg.save_videos:
                                image, wrist = get_libero_image(obs)
                                top_view.append(image)
                                wrist_view.append(wrist)
                            action = policy.get_action(obs, task.language)
                            obs, _, done, _ = env.step(action.tolist())
                            action_steps += 1
                            if done:
                                success = True
                                break
                        episode = {"initial_state_index": episode_idx, "success": success,
                                   "action_steps": action_steps}
                        if cfg.save_videos and top_view:
                            episode["video"] = save_rollout_video(
                                top_view, wrist_view, summary["total_episodes"] + 1,
                                success, description, log_file=log_file, output_dir=output / "videos")
                        result["episodes"].append(episode)
                        result["successes"] += int(success)
                        result["success_rate"] = result["successes"] / len(result["episodes"])
                        summary["total_episodes"] += 1
                        summary["total_successes"] += int(success)
                        save_summary()
                        message = (f"Task {task_id}, trial {episode_idx}: success={success}; "
                                   f"total {summary['total_successes']}/{summary['total_episodes']}")
                        print(message, flush=True)
                        log_file.write(message + "\n")
                        log_file.flush()
                finally:
                    if policy is not None:
                        policy.policy.close()
                    env.close()
        summary["status"] = "completed"
        save_summary()
        print(f"Success rate: {summary['success_rate']:.2%} "
              f"({summary['total_successes']}/{summary['total_episodes']})", flush=True)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        save_summary()
        raise


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
