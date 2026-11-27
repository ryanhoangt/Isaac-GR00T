"""Sequence-aware episode accounting around the original LIBERO-Mem domain.

The pinned fork can return done=True on an intermediate subgoal. We advance
its timers once before each action, retain its thresholds and overshoot logic,
and derive final success from the complete ordered sequence and final predicate.
"""


def _freeze(value):
    return tuple(_freeze(item) for item in value) if isinstance(value, (list, tuple)) else value


class LiberoMemEpisodeProtocol:
    def __init__(self, env):
        self.env = env
        self.domain = getattr(env, "env", env)
        required = (
            "get_all_goals",
            "get_goal_sequence_len",
            "reset_subgoal_progress",
            "_check_success",
            "_eval_predicate",
        )
        if any(not callable(getattr(self.domain, name, None)) for name in required):
            raise ValueError("Environment must use the LIBERO-Mem fork's sequence-aware domain")
        self.sequences = _freeze(self.domain.get_all_goals())
        lengths = {len(sequence) for sequence in self.sequences}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("Expected nonempty sequence alternatives with equal lengths")
        self.total = self.domain.get_goal_sequence_len()
        if self.total != next(iter(lengths)):
            raise ValueError("Goal sequence length disagrees with resolved BDDL alternatives")
        self.reset()

    def reset(self):
        self.domain.reset_subgoal_progress()
        # The upstream reset method leaves this flag set after an overshoot.
        self.domain._overshot = False
        self.action_steps = 0
        self.first_success_step = None

    def snapshot(self):
        """Read metrics without calling _check_success or changing its timers."""
        satisfied = self.domain._satisfied_subgoals
        if any(not clause or clause[0] != "and" for clause in satisfied):
            raise ValueError("Unexpected completed-subgoal structure")
        completed = _freeze([clause[1:] for clause in satisfied])
        count = len(completed)
        if count > self.total or not any(sequence[:count] == completed for sequence in self.sequences):
            raise ValueError("Completed subgoals are not an ordered prefix of any valid sequence")
        sequence_complete = count == self.total
        final_satisfied = sequence_complete and all(
            self.domain._eval_predicate(item) for item in satisfied[-1][1:]
        )
        overshot = bool(self.domain._overshot)
        return {
            "action_steps": self.action_steps,
            "subgoals_completed": count,
            "subgoals_total": self.total,
            "subgoal_completion_rate": count / self.total,
            "sequence_complete": sequence_complete,
            "final_goal_satisfied": bool(final_satisfied),
            "overshot": overshot,
            "success": bool(sequence_complete and final_satisfied and not overshot),
            "first_success_step": self.first_success_step,
        }

    def step(self, action):
        # Match mem_5/6's pre-action update, once per control step. env.step's
        # existing read checks use inc=False and remain part of the fork.
        self.domain._check_success(inc=True)
        obs, reward, done, info = self.env.step(action)
        self.action_steps += 1
        result = self.snapshot()
        if result["success"] and self.first_success_step is None:
            self.first_success_step = self.action_steps
            result["first_success_step"] = self.first_success_step
        result["environment_done"] = bool(done)
        result["reward"] = float(reward)
        return obs, result
