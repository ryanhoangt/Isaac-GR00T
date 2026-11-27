"""LIBERO-Mem episode initialization shared by expert and policy rollouts."""

import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env
from examples.Libero_Mem.episode import LiberoMemEpisodeProtocol


RESET_PROTOCOL = "libero_mem_compiled_xml_roundtrip_v1"


def _module_from_checkout(module, root):
    filename = getattr(module, "__file__", None)
    paths = [filename] if filename else list(getattr(module, "__path__", ()))
    return bool(paths) and all(Path(path).resolve().is_relative_to(root) for path in paths)


def initialize_episode(env, initial_state, *, num_steps_wait=20):
    """Match the benchmark's DeterministicInitWrapper before installing Raw state.

    Reload the freshly reset simulator's compiled XML and restore its state,
    then install the audited initial state and settle. Rebuilding from BDDL
    alone omits the benchmark's model round trip and can change replay physics.
    The caller owns the environment and must close it even if setup fails.
    """
    if type(num_steps_wait) is not int or num_steps_wait < 0:
        raise ValueError("num_steps_wait must be a nonnegative integer")
    protocol = LiberoMemEpisodeProtocol(env)
    env.reset()
    xml = env.sim.model.get_xml()
    state = env.sim.get_state().flatten().copy()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    if len(env.sim.get_state().flatten()) != len(initial_state):
        raise ValueError("Raw initial-state dimension differs from simulator model")
    obs = env.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    protocol.reset()
    return obs, protocol


def configure_runtime(config, output):
    """Use the fork and its customized robosuite in this process only."""
    repo = Path(config.libero_repo).expanduser().resolve(strict=True)
    for name, relative in (
        ("libero", "libero"),
        ("robosuite", "thirdparty/robosuite"),
        ("robomimic", "thirdparty/robomimic"),
    ):
        loaded = sys.modules.get(name)
        if loaded is not None and not _module_from_checkout(loaded, repo / relative):
            raise ValueError(f"{name} already imported from another checkout; start a fresh rollout process")
    sys.path[:0] = [str(repo), str(repo / "thirdparty/robosuite"), str(repo / "thirdparty/robomimic")]
    os.environ["MUJOCO_GL"] = config.renderer
    os.environ["PYOPENGL_PLATFORM"] = config.renderer
    os.environ.setdefault("LP_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    directory = output / ".libero"
    directory.mkdir()
    base = repo / "libero/libero"
    paths = {
        "benchmark_root": str(base),
        "assets": str(base / "assets"),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(repo / "scripts/__init_data"),
        "datasets": str(Path(config.raw_root).resolve()),
    }
    (directory / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    modules = {name: importlib.import_module(name) for name in ("mujoco", "robosuite", "libero.libero.envs")}
    expected = {"robosuite": repo / "thirdparty/robosuite", "libero.libero.envs": repo}
    for name, root in expected.items():
        if not Path(modules[name].__file__).resolve().is_relative_to(root):
            raise ValueError(f"Imported {name} from unexpected source")
    return {
        name: {"version": getattr(module, "__version__", None), "path": module.__file__}
        for name, module in modules.items()
    }


def make_environment(task, *, horizon, render, seed):
    descriptor = SimpleNamespace(
        language=task.instruction, problem_folder="libero_mem", bddl_file=f"{task.name}.bddl"
    )
    env, _ = get_libero_env(
        descriptor, horizon=horizon, use_camera_obs=render, camera_depths=False, camera_segmentations=None
    )
    env.seed(seed)
    actual = env.language_instruction.strip()
    if actual != task.instruction:
        env.close()
        raise ValueError("Simulator BDDL instruction differs from registry")
    return env
