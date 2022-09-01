from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Optional

Rect = NamedTuple("Rect", [("x", int), ("y", int), ("w", int), ("h", int)])


@dataclass
class GameState:
    score: int
    alive: bool
    dead: bool


class Action(IntEnum):
    NOTHING = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    @property
    def key(self) -> Optional[str]:
        return ACTION_TO_KEY[self]


ACTION_TO_KEY: dict[Action, Optional[str]] = {
    Action.NOTHING: None,
    Action.UP: "w",
    Action.DOWN: "s",
    Action.LEFT: "a",
    Action.RIGHT: "d",
}
