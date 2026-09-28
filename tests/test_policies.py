import json
from dataclasses import replace
from pathlib import Path
import time

import pytest

from yuri_next.contracts import Action, Observation
from yuri_next.policies import LayaPolicy, RulePolicy, create_policy
from yuri_next.runner import AsyncPolicyRunner


@pytest.fixture
def state():
    return Observation(**json.loads(Path("examples/observation.json").read_text()))


@pytest.mark.parametrize("changes,expected", [
    ({}, Action.ATTACK), ({"army_supply": 0}, Action.HOLD),
    ({"army_health": 0.2, "base_threat": 0.9}, Action.RETREAT),
    ({"base_threat": 0.6}, Action.DEFEND),
    ({"intel_age_seconds": 45}, Action.SCOUT),
    ({"army_supply": 10}, Action.HOLD),
])
def test_rule_priorities(state, changes, expected):
    assert RulePolicy().decide(replace(state, **changes)).action == expected


class RouterStub:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {"answers": {"action": self.answer}}


def test_laya_contract_and_reuse(state):
    router = RouterStub({"choice": "attack", "answer_confidence": 0.9})
    policy = LayaPolicy(router_factory=lambda: router)
    assert policy.decide(state).executed_policy == "laya"
    assert policy.decide(state).confidence == 0.9
    assert len(router.calls) == 2
    assert router.calls[0][1]["action"]["type"] == "choice"


@pytest.mark.parametrize("answer", [
    {"choice": "attack", "answer_confidence": 0.1},
    {"choice": "attack"}, {"choice": "cheat", "answer_confidence": 0.9},
    {"choice": "attack", "answer_confidence": float("nan")},
    {"choice": "attack", "answer_confidence": 0.99, "low_confidence": True},
])
def test_laya_bad_answers_fall_back(state, answer):
    policy = LayaPolicy(router_factory=lambda: RouterStub(answer))
    decision = policy.decide(state)
    assert decision.requested_policy == "laya"
    assert decision.executed_policy == "rules"
    assert decision.fallback_reason
    assert decision.confidence is None


def test_illegal_action_and_mask(state):
    router = RouterStub({"choice": "attack", "answer_confidence": 0.9})
    decision = LayaPolicy(router_factory=lambda: router).decide(replace(state, army_supply=0))
    assert decision.action == Action.HOLD
    assert decision.executed_policy == "rules"
    assert list(router.calls[0][1]["action"]["criteria"]) == ["hold"]


def test_load_failure(state):
    def fail():
        raise RuntimeError("model unavailable")
    decision = LayaPolicy(router_factory=fail).decide(state)
    assert "model unavailable" in decision.fallback_reason


def test_late_result(state):
    class SlowRouter(RouterStub):
        def predict(self, *args, **kwargs):
            time.sleep(0.01)
            return super().predict(*args, **kwargs)
    result = LayaPolicy(budget_ms=1, router_factory=lambda: SlowRouter(
        {"choice": "attack", "answer_confidence": 0.9})).decide(state)
    assert "decision_budget_exceeded" in result.fallback_reason


def test_runner_does_not_queue_unbounded_work(state):
    from threading import Event
    ready, release = Event(), Event()
    class BlockingPolicy:
        def decide(self, observation):
            ready.set()
            release.wait(2)
            return RulePolicy().decide(observation)
    runner = AsyncPolicyRunner(BlockingPolicy())
    try:
        runner.tick(state)
        assert ready.wait(1)
        future = runner.pending
        for _ in range(20):
            assert runner.tick(state).fallback_reason
            assert runner.pending is future
    finally:
        release.set()
        runner.close()


@pytest.mark.parametrize("changes", [{"army_supply": -1}, {"army_health": 1.1},
    {"army_supply": float("nan")}, {"schema_version": 2}, {"enemy_visible": "yes"}])
def test_invalid_observations(state, changes):
    with pytest.raises(ValueError):
        replace(state, **changes)


def test_unknown_policy():
    with pytest.raises(ValueError):
        create_policy("unknown")


def test_shared_browser_fixtures():
    cases = json.loads(Path("tests/fixtures/policy-cases.json").read_text())
    for case in cases:
        assert RulePolicy().decide(Observation(**case["observation"])).action == case["action"], case["name"]


@pytest.mark.parametrize("changes", [{"base_threat": .9}, {"game_time_seconds": 500}])
def test_runner_rejects_urgent_or_old_game_state(state, changes):
    from concurrent.futures import Future
    from time import monotonic
    runner = AsyncPolicyRunner(RulePolicy())
    ready = Future()
    ready.set_result(RulePolicy().decide(state))
    runner.pending, runner.started, runner.source_observation = ready, monotonic(), state
    try:
        result = runner.tick(replace(state, **changes))
        assert result.fallback_reason == "pending_stale_or_urgent"
        assert runner.accepted_observation is None
    finally:
        runner.close()
