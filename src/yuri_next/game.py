"""Experimental burnysc2 adapter, deliberately independent of legacy bots."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import re

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.main import run_game
from sc2.player import Bot, Computer

from .contracts import Action, Observation, legal_actions
from .policies import create_policy
from .runner import AsyncPolicyRunner


class YuriBot(BotAI):
    ENEMY_MEMORY_SECONDS = 120
    ACTION_HYSTERESIS_SECONDS = 6

    def __init__(self, policy_name="rules", *, map_name="AcropolisLE",
                 difficulty="medium", random_seed=1, run_id=None):
        super().__init__()
        self.policy_name = policy_name
        self.run_id = run_id
        self.game_metadata = {"policy": policy_name, "map": map_name,
                              "difficulty": difficulty, "random_seed": random_seed,
                              "run_id": run_id}
        self.policy = create_policy(policy_name)
        self.runner = AsyncPolicyRunner(self.policy) if policy_name == "laya" else None
        self.last_decision = -10
        self.last_macro = -10
        self.enemy_memory = {}
        self.current_action = None
        self.action_since = 0
        self.trace_file = None
        self.trace_path = None

    async def on_start(self):
        directory = Path("artifacts/games")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.trace_path = directory / f"{stamp}-{self.policy_name}.jsonl"
        self.game_metadata["started_at"] = datetime.now(timezone.utc).isoformat()
        self.trace_file = self.trace_path.open("w", encoding="utf-8")

    @property
    def army(self):
        return self.units.of_type({U.STALKER, U.VOIDRAY, U.ZEALOT})

    def observe(self):
        army = self.army
        enemies = self.enemy_units.filter(lambda u: u.type_id not in {U.PROBE, U.SCV, U.DRONE, U.MULE} and u.can_attack)
        for enemy in enemies:
            self.enemy_memory[enemy.tag] = (self.calculate_supply_cost(enemy.type_id), self.time)
        self.enemy_memory = {
            tag: memory for tag, memory in self.enemy_memory.items()
            if self.time - memory[1] <= self.ENEMY_MEMORY_SECONDS
        }
        known_enemy_supply = sum(supply for supply, _seen_at in self.enemy_memory.values())
        latest_intel = max((seen_at for _supply, seen_at in self.enemy_memory.values()), default=0)
        nearby = enemies.filter(lambda e: any(e.distance_to(b) < 22 for b in self.townhalls))
        health = sum(u.health + u.shield for u in army)
        maximum = sum(u.health_max + u.shield_max for u in army)
        return Observation(
            army_supply=sum(self.calculate_supply_cost(u.type_id) for u in army),
            enemy_army_supply=known_enemy_supply,
            base_threat=min(1, sum(self.calculate_supply_cost(u.type_id) for u in nearby) / 12),
            army_health=health / maximum if maximum else 1,
            intel_age_seconds=max(0, self.time - latest_intel),
            minerals=self.minerals, gas=self.vespene, supply_left=max(0, self.supply_left),
            game_time_seconds=self.time, enemy_visible=bool(enemies),
        )

    async def on_step(self, iteration):
        if self.time - self.last_macro >= 1:
            self.last_macro = self.time
            await self.distribute_workers()
            await self.macro()
        if self.time - self.last_decision < 2:
            return
        self.last_decision = self.time
        observation = self.observe()
        decision = self.runner.tick(observation) if self.runner else self.policy.decide(observation)
        decision = self.apply_hysteresis(decision, observation)
        self.execute(decision.action)
        record = {"schema_version": 1, "source": "sc2-runtime",
                  "observation": observation.to_dict(), "decision": decision.to_dict()}
        if self.runner and self.runner.accepted_observation is not None:
            record["inference_observation"] = self.runner.accepted_observation.to_dict()
        self.trace_file.write(json.dumps(record) + "\n")
        self.trace_file.flush()

    def apply_hysteresis(self, decision, observation):
        proposed = decision.action
        urgent = (
            observation.army_supply <= 0
            or observation.army_health < 0.35
            or observation.base_threat >= 0.6
            or proposed not in legal_actions(observation)
        )
        if self.current_action is None:
            self.current_action = proposed
            self.action_since = observation.game_time_seconds
        elif proposed != self.current_action:
            if urgent or observation.game_time_seconds - self.action_since >= self.ACTION_HYSTERESIS_SECONDS:
                self.current_action = proposed
                self.action_since = observation.game_time_seconds
            else:
                return replace(
                    decision,
                    action=self.current_action,
                    proposed_action=proposed,
                    reason=f"Action held for stability; policy proposed {proposed.value}.",
                )
        return decision

    async def macro(self):
        # One spending action per tick prevents multiple commands overspending
        # the same mineral snapshot. This is a baseline, not a tuned build order.
        bases = self.townhalls.ready
        if not bases:
            return
        if self.supply_cap < 200 and self.supply_left < 5 and not self.already_pending(U.PYLON) and self.can_afford(U.PYLON):
            await self.build(U.PYLON, near=bases.first.position.towards(self.game_info.map_center, 5))
            return
        worker_target = min(
            65,
            self.townhalls.amount * 18 + min(self.gas_buildings.amount * 3, self.townhalls.amount * 6),
        )
        if self.workers.amount + self.already_pending(U.PROBE) < worker_target:
            for base in bases.idle:
                if self.can_afford(U.PROBE) and self.supply_left > 0:
                    base.train(U.PROBE)
                    return
        if (self.townhalls.amount < 3 and self.minerals >= 400
                and self.workers.amount >= 16 * self.townhalls.amount
                and not self.already_pending(U.NEXUS)):
            await self.expand_now()
            return
        gas_target = min(6, self.townhalls.amount * 2)
        if self.gas_buildings.amount + self.already_pending(U.ASSIMILATOR) < gas_target:
            for base in bases:
                for geyser in self.vespene_geyser.closer_than(10, base):
                    if not self.gas_buildings.closer_than(1, geyser) and not self.already_pending(U.ASSIMILATOR):
                        worker = self.select_build_worker(geyser.position)
                        if worker and self.can_afford(U.ASSIMILATOR):
                            worker.build_gas(geyser)
                            return
        pylons = self.structures(U.PYLON).ready
        if not pylons:
            return
        for building, prerequisite in ((U.GATEWAY, U.PYLON), (U.CYBERNETICSCORE, U.GATEWAY)):
            if not self.structures(building) and not self.already_pending(building):
                if self.structures(prerequisite).ready and self.can_afford(building):
                    await self.build(building, near=pylons.first)
                    return
        stargate_goal = min(2, self.townhalls.amount)
        if (self.structures(U.CYBERNETICSCORE).ready
                and self.structures(U.STARGATE).amount + self.already_pending(U.STARGATE) < stargate_goal
                and self.can_afford(U.STARGATE)):
            await self.build(U.STARGATE, near=pylons.first)
            return
        if self.structures(U.CYBERNETICSCORE).ready:
            for building, unit in ((U.STARGATE, U.VOIDRAY), (U.GATEWAY, U.STALKER)):
                for producer in self.structures(building).ready.idle:
                    if self.can_afford(unit) and self.supply_left >= self.calculate_supply_cost(unit):
                        producer.train(unit)
                        return
    def execute(self, action):
        army = self.army
        if not army:
            return
        home = self.townhalls.first.position if self.townhalls else self.start_location
        if action == Action.SCOUT:
            army.closest_to(self.enemy_start_locations[0]).move(self.enemy_start_locations[0])
            return
        if action == Action.RETREAT:
            for unit in army:
                unit.move(home)
            return
        if action == Action.DEFEND:
            # Select the most threatened base, rather than always the main.
            if self.townhalls and self.enemy_units:
                combat_enemies = self.enemy_units.filter(lambda u: u.can_attack)
                base = max(self.townhalls, key=lambda b: sum(
                    self.calculate_supply_cost(enemy.type_id) / (1 + enemy.distance_to(b))
                    for enemy in combat_enemies if enemy.distance_to(b) < 25))
                home = base.position
            threatened = self.enemy_units.filter(lambda u: u.can_attack and u.distance_to(home) < 25)
            target = threatened.closest_to(home).position if threatened else home
        elif action == Action.ATTACK:
            target = self.enemy_structures.first.position if self.enemy_structures else self.enemy_start_locations[0]
        else:
            target = home.towards(self.game_info.map_center, 7)
        for unit in army:
            unit.attack(target)

    async def on_end(self, game_result):
        if self.runner:
            self.runner.close()
        if self.trace_file:
            self.trace_file.close()
            metadata = {
                "schema_version": 1,
                "source": "sc2-game-metadata",
                **self.game_metadata,
                "result": getattr(game_result, "name", str(game_result)),
                "game_time_seconds": self.time,
                "trace": str(self.trace_path),
            }
            self.trace_path.with_suffix(".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


def launch(policy, map_name, difficulty, realtime, random_seed=1, run_id=None):
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", run_id):
        raise ValueError("run_id may contain only letters, numbers, dots, underscores, and dashes")
    level = {"easy": Difficulty.Easy, "medium": Difficulty.Medium, "hard": Difficulty.Hard}[difficulty]
    from sc2.paths import Paths
    bot = YuriBot(policy, map_name=map_name, difficulty=difficulty,
                  random_seed=random_seed, run_id=run_id)
    bot.game_metadata["sc2_build"] = Paths.EXECUTABLE.parent.name
    return run_game(
        maps.get(map_name), [Bot(Race.Protoss, bot), Computer(Race.Zerg, level)],
        realtime=realtime, random_seed=random_seed,
    )
