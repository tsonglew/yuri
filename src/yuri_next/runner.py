"""Non-blocking single-flight inference for a live game loop."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import monotonic

from .contracts import Observation, legal_actions
from .policies import RulePolicy


class AsyncPolicyRunner:
    def __init__(self, policy, *, max_age_seconds=3):
        self.policy = policy
        self.max_age_seconds = max_age_seconds
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yuri-policy")
        self.pending = None
        self.started = 0
        self.source_observation = None
        self.accepted_observation = None
        self.rules = RulePolicy()

    def tick(self, observation: Observation):
        decision = None
        self.accepted_observation = None
        if self.pending is not None and self.pending.done():
            try:
                candidate = self.pending.result()
                game_age = observation.game_time_seconds - self.source_observation.game_time_seconds
                if monotonic() - self.started <= self.max_age_seconds and 0 <= game_age <= 6:
                    decision = candidate
                    self.accepted_observation = self.source_observation
            except Exception:
                decision = None
            self.pending = None
        # Urgent events and currently illegal actions always override old results.
        urgent = observation.army_health < 0.35 or observation.base_threat >= 0.6
        if decision is not None and (urgent or decision.action not in legal_actions(observation)):
            decision = None
        if self.pending is None:
            self.started = monotonic()
            self.source_observation = observation
            self.pending = self.pool.submit(self.policy.decide, observation)
        if decision is None:
            self.accepted_observation = None
            return replace(self.rules.decide(observation), requested_policy="laya",
                           fallback_reason="pending_stale_or_urgent")
        return decision

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
