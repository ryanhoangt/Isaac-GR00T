"""Reactive GR00T policy using LIBERO's shared adapter and HTTP transport."""

from collections import deque

from examples.Libero.adapter import to_gr00t_observation, to_libero_actions
from gr00t.eval.http_client import HTTPInferenceClient


class LiberoMemPolicy:
    def __init__(self, server_url="http://127.0.0.1:5555", *, timeout=60.0, n_action_steps=1, client=None):
        if type(n_action_steps) is not int or n_action_steps < 1:
            raise ValueError("n_action_steps must be a positive integer")
        self.client = client if client is not None else HTTPInferenceClient(server_url, timeout=timeout)
        self.n_action_steps = n_action_steps
        self.reset()

    def reset(self):
        # /act is stateless and has no reset endpoint. Drop all old queued actions.
        self._actions = deque()
        self.requests = 0
        self.prediction_horizon = None
        self._language = None

    def get_action(self, observation, language):
        if self._language is not None and language != self._language:
            raise ValueError("Instruction changed within an episode; call reset() first")
        self._language = language
        if not self._actions:
            prediction = self.client.get_action(to_gr00t_observation(observation, language))
            self.requests += 1
            actions = to_libero_actions(prediction)
            if len(actions) < self.n_action_steps:
                raise ValueError("n_action_steps exceeds the server's prediction horizon")
            self.prediction_horizon = len(actions)
            self._actions.extend(actions[: self.n_action_steps])
        return self._actions.popleft().copy()

    def close(self):
        self._actions.clear()
        self.client.close()
