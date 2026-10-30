from __future__ import annotations
from collections import deque

import os
import json
import pprint
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

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
    # fmt: off
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    #################################################################################################################
    # fmt: on
    """Port to connect to."""
    port: int = 8071
    """Headless mode (no GUI)."""
    headless: bool = True
    run_name: str = "default"
    output_dir: Optional[str] = None
    """Override the directory containing rollout videos, results.log, and summary.json."""
    host: str = "localhost"
    task_ids: Optional[List[int]] = None
    """Evaluate only these task IDs; by default evaluate the entire suite."""
    inference_timeout_ms: int = 15000
    """RPC timeout; shared-server evaluations can allow time for queued requests."""
    record_uctx: bool = False
    context_diagnostics: bool = False
    save_observations: bool = False


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

    def __init__(self, host="localhost", port=5555, headless=False, inference_timeout_ms=15000,
                 recording_root=None, context_diagnostics=False, save_observations=False):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port, timeout_ms=inference_timeout_ms)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless
        self.actions = deque()
        self.recording_root = recording_root
        self.context_diagnostics = context_diagnostics
        self.save_observations = save_observations
        self.recorder = None
        self.control_step = 0
        self.score_signature = (self.policy.call_endpoint("get_score_signature", requires_input=False)
                                if recording_root is not None else None)

    def reset(self, task_id=None, episode_index=None):
        self.actions.clear()
        self.control_step = 0
        if self.recording_root is not None:
            from pathlib import Path
            from examples.Libero.eval.uncertainty_recording import EpisodeUncertaintyRecorder
            if task_id is None or episode_index is None:
                raise ValueError("recording requires task and episode IDs")
            self.recorder = EpisodeUncertaintyRecorder(
                Path(self.recording_root) / f"task{task_id:02d}_init{episode_index:03d}",
                task_id, episode_index, self.score_signature, self.save_observations,
            )

    def get_action(self, observation_dict, lang: str):
        """Get action from GR00T policy given observation and language instruction."""
        if len(self.actions) == 0:
            obs_dict = self._process_observation(observation_dict, lang)
            # summarize_obs(obs_dict)
            if self.recording_root is None:
                result = None
                action_chunk = self.policy.get_action(obs_dict)
            else:
                endpoint = ("get_action_with_context_diagnostics" if self.context_diagnostics
                            else "get_action_with_uncertainty")
                result = self.policy.call_endpoint(endpoint, obs_dict)
                action_chunk = result["action"]
            actions = self._convert_to_libero_action(action_chunk)
            if self.recorder is not None:
                self.recorder.record(self.control_step, obs_dict, result, actions)
            self.actions.extend(actions)
        self.control_step += 1
        return self.actions.popleft()

    def finish_episode(self, success):
        if self.recorder is not None:
            self.recorder.finish(success, self.control_step)

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
        # action_components = [
        #     np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        # ]
        # action_array = np.array(action_components, dtype=np.float32)
        action_array = np.concatenate([action_chunk[f"action.{key}"] for key in self.action_keys], axis=1)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert action_array.shape[-1] == 7, f"Expected 7-dim action, got {action_array.shape[-1]}"
        return action_array


def eval_libero(cfg: GenerateConfig) -> None:
    if (cfg.context_diagnostics or cfg.save_observations) and not cfg.record_uctx:
        raise ValueError("context diagnostics / raw observation saving require --record-uctx")
    log_dir = cfg.output_dir or f"./runs/gr00t/{cfg.task_suite_name}/{cfg.run_name}"
    os.makedirs(log_dir, exist_ok=True)  # ensures directory exists

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = list(range(num_tasks_in_suite)) if cfg.task_ids is None else cfg.task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be nonempty and contain no duplicates")
    if any(task_id < 0 or task_id >= num_tasks_in_suite for task_id in task_ids):
        raise ValueError(f"task_ids must be between 0 and {num_tasks_in_suite - 1}")
    if cfg.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    print(f"Task suite: {cfg.task_suite_name}")
    log_file = open(f"{log_dir}/results.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    episode_results = []
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states):
            raise ValueError("Requested more trials than available LIBERO initial states")

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        gr00t_policy = GR00TPolicy(
            host=cfg.host, port=cfg.port, headless=cfg.headless,
            inference_timeout_ms=cfg.inference_timeout_ms,
            recording_root=os.path.join(log_dir, "uncertainty") if cfg.record_uctx else None,
            context_diagnostics=cfg.context_diagnostics, save_observations=cfg.save_observations,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            gr00t_policy.reset(task_id=task_id, episode_index=episode_idx)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            top_view = []
            wrist_view = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 600  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 1000  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # # Get preprocessed image
                    img, wrist_img = get_libero_image(obs)

                    # # Save preprocessed image for replay video
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    # Query model to get action
                    action = gr00t_policy.get_action(
                        obs,
                        task.language,
                    )

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    log_file.flush()
                    raise

            task_episodes += 1
            total_episodes += 1
            gr00t_policy.finish_episode(done)
            episode_results.append({
                "task_id": task_id,
                "task_description": task_description,
                "episode_index": episode_idx,
                "success": bool(done),
                "control_steps": t - cfg.num_steps_wait + int(bool(done)),
            })

            # Save a replay video of the episode
            save_rollout_video(
                top_view,
                wrist_view,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                log_dir=log_dir,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()
        env.close()

        with open(f"{log_dir}/summary.json", "w") as summary_file:
            json.dump({
                "task_suite_name": cfg.task_suite_name,
                "run_name": cfg.run_name,
                "task_ids": task_ids,
                "num_trials_per_task": cfg.num_trials_per_task,
                "num_episodes": total_episodes,
                "num_successes": total_successes,
                "success_rate": total_successes / total_episodes,
                "episodes": episode_results,
            }, summary_file, indent=2)

    # Save local log file
    log_file.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
