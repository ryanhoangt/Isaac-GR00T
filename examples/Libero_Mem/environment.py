"""LIBERO-Mem episode initialization shared by expert and policy rollouts."""

from examples.Libero.eval.utils import get_libero_dummy_action
from examples.Libero_Mem.episode import LiberoMemEpisodeProtocol


RESET_PROTOCOL = "libero_mem_compiled_xml_roundtrip_v1"


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
