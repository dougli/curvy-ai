from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Optional

Rect = NamedTuple("Rect", [("x", int), ("y", int), ("w", int), ("h", int)])


@dataclass
class Player:
    score: int
    alive: bool
    dead: bool


class Action(IntEnum):
    NOTHING = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

    @property
    def key(self) -> Optional[str]:
        return ACTION_TO_KEY[self]


ACTION_TO_KEY: dict[Action, Optional[str]] = {
    Action.NOTHING: None,
    Action.LEFT: "a",
    Action.RIGHT: "d",
    Action.UP: "w",
    Action.DOWN: "s",
}


@dataclass
class Account:
    email: str
    password: str
    match_name: str
    match_password: str
    is_host: bool
    wait_for: tuple[str, ...] = tuple()
