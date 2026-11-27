"""Canonical task identity and the proposal's complementary training folds.

Names follow libero-mem commit 0eee0defa4024e51511b6e15a79f30e9620209de,
libero/libero/benchmark/libero_suite_task_map.py. Importing this registry does
not import the simulator, Torch, or any model dependencies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    number: int
    instruction: str
    category: str
    memory_types: tuple[str, ...]
    fold: str

    @property
    def name(self) -> str:
        return f"KITCHEN_SCENE1_{self.number}_{self.instruction.replace(' ', '_')}"

    @property
    def filename(self) -> str:
        return f"{self.name}_demo.hdf5"


# category is the proposal's reporting grouping. memory_types are the original
# benchmark's Object Motion / Sequence / Relations / Occlusion (OM/OS/OR/OO).
TASKS = (
    Task(1, "pick up the bowl and place it back on the plate", "single", ("OM",), "A"),
    Task(2, "lift the bottle and put it down on the plate", "single", ("OM",), "B"),
    Task(3, "lift the bowl and place it back on the plate 3 times", "counting", ("OM", "OS"), "A"),
    Task(4, "pick up the bottle and put it down the plate 3 times", "counting", ("OM", "OS"), "B"),
    Task(5, "lift the bowl and place it back on the plate 5 times", "counting", ("OM", "OS"), "A"),
    Task(6, "pick up the bowl and place it on the plate 7 times", "counting", ("OM", "OS"), "B"),
    Task(7, "swap the 2 bowls on their plates using the empty plate", "sequencing_spatial", ("OM", "OR"), "A"),
    Task(
        8,
        "rotate the 3 bowls on their plates from left to right using the empty plate",
        "sequencing_spatial",
        ("OM", "OR"),
        "B",
    ),
    Task(
        9,
        "put the cream cheese in the nearest basket and place that basket in the center",
        "sequencing_spatial",
        ("OM", "OO"),
        "A",
    ),
    Task(
        10,
        "put the cream cheese in the nearest basket and place the empty basket in the center",
        "sequencing_spatial",
        ("OM", "OO"),
        "B",
    ),
)
TASK_BY_NUMBER = {task.number: task for task in TASKS}
TASK_BY_NAME = {task.name: task for task in TASKS}


def get_fold(training_fold: str) -> dict[str, list[str]]:
    """A means train A / test B; B means train B / test A. These are task splits."""
    if training_fold not in ("A", "B"):
        raise ValueError("training_fold must be A or B")
    return {
        "train_tasks": [task.name for task in TASKS if task.fold == training_fold],
        "held_out_tasks": [task.name for task in TASKS if task.fold != training_fold],
    }
