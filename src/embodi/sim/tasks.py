from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    id: str
    instruction: str
    pretraining: bool
    success_predicate: str
    dwell_steps: int
    requires_release: bool

    def __post_init__(self) -> None:
        if not self.id or not self.instruction or not self.success_predicate:
            raise ValueError("task id, instruction, and success predicate must not be empty")
        if self.dwell_steps <= 0:
            raise ValueError("task dwell_steps must be positive")


@dataclass(frozen=True)
class TaskStatus:
    success: bool
    terminal: bool
    stage: str
    failure_reason: str | None = None


PUSH_TO_ZONE = TaskSpec(
    id="push_to_zone",
    instruction="Push the object into the target zone.",
    pretraining=True,
    success_predicate="object_center_inside_target",
    dwell_steps=15,
    requires_release=False,
)
LIFT_OBJECT = TaskSpec(
    id="lift_object",
    instruction="Lift the object and hold it above the table.",
    pretraining=True,
    success_predicate="object_above_height_while_retained",
    dwell_steps=15,
    requires_release=False,
)
PICK_PLACE_BIN = TaskSpec(
    id="pick_place_bin",
    instruction="Pick up the object and place it in the bin.",
    pretraining=True,
    success_predicate="object_contained_and_settled",
    dwell_steps=15,
    requires_release=True,
)
STACK_OBJECT = TaskSpec(
    id="stack_object",
    instruction="Stack the object stably on the support object.",
    pretraining=False,
    success_predicate="object_supported_and_settled",
    dwell_steps=15,
    requires_release=True,
)

TASKS = {task.id: task for task in (PUSH_TO_ZONE, LIFT_OBJECT, PICK_PLACE_BIN, STACK_OBJECT)}


def task_by_id(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as error:
        raise ValueError(f"unknown task: {task_id}") from error
