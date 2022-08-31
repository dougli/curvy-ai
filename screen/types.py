from dataclasses import dataclass


@dataclass
class GameState:
    score: int
    alive: bool
    dead: bool
