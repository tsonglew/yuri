"""Versioned, JSON-safe contracts shared by policies and decision replays."""

from dataclasses import asdict, dataclass
from enum import StrEnum
import math
from typing import Protocol


class Action(StrEnum):
    HOLD = "hold"
    DEFEND = "defend"
    ATTACK = "attack"
    RETREAT = "retreat"
    SCOUT = "scout"


@dataclass(frozen=True)
class Observation:
    army_supply: float
    enemy_army_supply: float
    base_threat: float
    army_health: float
    intel_age_seconds: float
    minerals: int = 0
    gas: int = 0
    supply_left: float = 0
    game_time_seconds: float = 0
    enemy_visible: bool = False
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported observation schema_version")
        if type(self.enemy_visible) is not bool:
            raise ValueError("enemy_visible must be a boolean")
        for key, value in asdict(self).items():
            if key in ("enemy_visible", "schema_version"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.base_threat > 1 or self.army_health > 1:
            raise ValueError("base_threat and army_health must be in [0, 1]")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    action: Action
    requested_policy: str
    executed_policy: str
    reason: str
    latency_ms: float = 0
    confidence: float | None = None
    fallback_reason: str | None = None
    model: str | None = None
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


class Policy(Protocol):
    def decide(self, observation: Observation) -> Decision: ...


def legal_actions(observation: Observation) -> set[Action]:
    # Scouting in the initial executor uses combat units, never a worker.
    if observation.army_supply <= 0:
        return {Action.HOLD}
    return set(Action)
