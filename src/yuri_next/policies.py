"""Policy implementations; importing this module does not load ML libraries."""

from dataclasses import replace
import json
import math
import os
from time import perf_counter
from typing import Callable

from .contracts import Action, Decision, Observation, Policy, legal_actions


class RulePolicy:
    def decide(self, observation: Observation) -> Decision:
        start = perf_counter()
        o = observation
        if o.army_supply <= 0:
            action, reason = Action.HOLD, "No combat army available."
        elif o.army_health < 0.35:
            action, reason = Action.RETREAT, "Army health below 35%."
        elif o.base_threat >= 0.6:
            action, reason = Action.DEFEND, "Base threat at or above 60%."
        elif o.intel_age_seconds >= 45:
            action, reason = Action.SCOUT, "Enemy intelligence is at least 45 seconds old."
        elif o.army_supply >= 12 and o.army_supply >= 1.3 * max(o.enemy_army_supply, 1):
            action, reason = Action.ATTACK, "At least 12 army supply and a 1.3x observed advantage."
        else:
            action, reason = Action.HOLD, "Rally and build strength."
        return Decision(action, "rules", "rules", reason, (perf_counter() - start) * 1000)


class LayaPolicy:
    """Lazy real Router integration, with explicit and observable rule fallback.

    Call from a worker, not the SC2 event loop. A time budget here rejects late
    results; it cannot interrupt a native GPU call. AsyncPolicyRunner ensures
    at most one inference is in flight for live play.
    """

    def __init__(self, *, min_confidence=0.65, budget_ms=1500,
                 model="english", revision: str | None = None,
                 router_factory: Callable | None = None):
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1 or not math.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError("Invalid confidence threshold or time budget")
        self.min_confidence = min_confidence
        self.budget_ms = budget_ms
        self.model = model
        self.revision = revision if revision is not None else os.environ.get("YURI_LAYA_REVISION")
        self._factory = router_factory
        self._router = None
        self.fallback = RulePolicy()

    def _get_router(self):
        if self._router is None:
            if self._factory is None:
                from laya import Router
                self._router = Router(revision=self.revision)
            else:
                self._router = self._factory()
        return self._router

    def decide(self, observation: Observation) -> Decision:
        start = perf_counter()
        try:
            allowed = legal_actions(observation)
            criteria = {
                "hold": "Rally at our base and build strength; no attack order.",
                "defend": "Defend our base against an immediate enemy threat.",
                "attack": "Attack the enemy with a healthy superior army and recent intelligence.",
                "retreat": "Move the army back to safety when health or strength is insufficient.",
                "scout": "Send one combat unit to refresh stale enemy intelligence.",
            }
            questions = {"action": {
                "type": "choice",
                "instructions": "Choose one StarCraft II macro action. Enemy supply is a last-seen estimate, not a guaranteed current count or total enemy strength. Prefer survival when uncertain.",
                "criteria": {k: v for k, v in criteria.items() if Action(k) in allowed},
            }}
            result = self._get_router().predict(
                json.dumps(observation.to_dict(), sort_keys=True), questions,
                model=self.model, min_confidence=self.min_confidence,
            )
            answer = result["answers"]["action"]
            action = Action(answer["choice"])
            confidence = answer.get("answer_confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                raise ValueError("missing_confidence")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid_confidence")
            if answer.get("low_confidence") or confidence < self.min_confidence:
                raise ValueError("low_confidence")
            if action not in allowed:
                raise ValueError("illegal_action")
            elapsed = (perf_counter() - start) * 1000
            if elapsed > self.budget_ms:
                raise TimeoutError("decision_budget_exceeded")
            revisions = getattr(self._router, "loaded_revisions", {})
            model_revision = revisions.get(self.model) if isinstance(revisions, dict) else None
            return Decision(action, "laya", "laya", "Selected by Laya; no generated explanation.",
                            elapsed, confidence, model=self.model,
                            model_revision=model_revision)
        except Exception as error:
            fallback = self.fallback.decide(observation)
            return replace(fallback, requested_policy="laya", model=self.model,
                           latency_ms=(perf_counter() - start) * 1000,
                           fallback_reason=f"{type(error).__name__}: {error}")


_POLICIES: dict[str, Callable[..., Policy]] = {"rules": RulePolicy, "laya": LayaPolicy}


def register_policy(name: str, factory: Callable[..., Policy]):
    if not name or name in _POLICIES:
        raise ValueError(f"Policy already registered or invalid: {name}")
    _POLICIES[name] = factory


def create_policy(name: str, **kwargs) -> Policy:
    try:
        factory = _POLICIES[name]
    except KeyError:
        raise ValueError(f"Unknown policy: {name}; available: {', '.join(_POLICIES)}") from None
    return factory(**kwargs)
