"""Experimental burnysc2 adapter, deliberately independent of legacy bots."""

import json
from pathlib import Path
from datetime import datetime, timezone

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.main import run_game
from sc2.player import Bot, Computer

from .contracts import Action, Observation
from .policies import create_policy
from .runner import AsyncPolicyRunner


class YuriBot(BotAI):
    def __init__(self, policy_name="rules"):
        super().__init__()
        self.policy_name = policy_name
        self.policy = create_policy(policy_name)
        self.runner = AsyncPolicyRunner(self.policy) if policy_name == "laya" else None
        self.last_decision = -10
        self.last_macro = -10
        self.last_intel = 0
        self.observed_enemy_supply = 0
        self.trace_file = None

    async def on_start(self):
        directory = Path("artifacts/games")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.trace_file = (directory / f"{stamp}-{self.policy_name}.jsonl").open("w", encoding="utf-8")

    @property
    def army(self):
        return self.units.of_type({U.STALKER, U.VOIDRAY, U.ZEALOT})

    def observe(self):
        army = self.army
        enemies = self.enemy_units.filter(lambda u: u.type_id not in {U.PROBE, U.SCV, U.DRONE, U.MULE} and u.can_attack)
        if enemies:
            self.last_intel = self.time
            self.observed_enemy_supply = sum(self.calculate_supply_cost(u.type_id) for u in enemies)
        nearby = enemies.filter(lambda e: any(e.distance_to(b) < 22 for b in self.townhalls))
        health = sum(u.health + u.shield for u in army)
        maximum = sum(u.health_max + u.shield_max for u in army)
        return Observation(
            army_supply=sum(self.calculate_supply_cost(u.type_id) for u in army),
            enemy_army_supply=self.observed_enemy_supply,
            base_threat=min(1, sum(self.calculate_supply_cost(u.type_id) for u in nearby) / 12),
            army_health=health / maximum if maximum else 1,
            intel_age_seconds=max(0, self.time - self.last_intel),
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
        self.execute(decision.action)
        record = {"schema_version": 1, "source": "sc2-runtime",
                  "observation": observation.to_dict(), "decision": decision.to_dict()}
        if self.runner and self.runner.accepted_observation is not None:
            record["inference_observation"] = self.runner.accepted_observation.to_dict()
        self.trace_file.write(json.dumps(record) + "\n")
        self.trace_file.flush()

    async def macro(self):
        # One spending action per tick prevents multiple commands overspending
        # the same mineral snapshot. This is a baseline, not a tuned build order.
        bases = self.townhalls.ready
        if not bases:
            return
        if self.supply_cap < 200 and self.supply_left < 5 and not self.already_pending(U.PYLON) and self.can_afford(U.PYLON):
            await self.build(U.PYLON, near=bases.first.position.towards(self.game_info.map_center, 5))
            return
        if self.workers.amount + self.already_pending(U.PROBE) < min(65, self.townhalls.amount * 22):
            for base in bases.idle:
                if self.can_afford(U.PROBE) and self.supply_left > 0:
                    base.train(U.PROBE)
                    return
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
        for building, prerequisite in ((U.GATEWAY, U.PYLON), (U.CYBERNETICSCORE, U.GATEWAY), (U.STARGATE, U.CYBERNETICSCORE)):
            if not self.structures(building) and not self.already_pending(building):
                if self.structures(prerequisite).ready and self.can_afford(building):
                    await self.build(building, near=pylons.first)
                    return
        if self.structures(U.CYBERNETICSCORE).ready:
            for building, unit in ((U.STARGATE, U.VOIDRAY), (U.GATEWAY, U.STALKER)):
                for producer in self.structures(building).ready.idle:
                    if self.can_afford(unit) and self.supply_left >= self.calculate_supply_cost(unit):
                        producer.train(unit)
                        return
        if self.townhalls.amount < 3 and not self.already_pending(U.NEXUS) and self.can_afford(U.NEXUS):
            await self.expand_now()

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
                base = max(self.townhalls, key=lambda b: sum(
                    1 for enemy in self.enemy_units if enemy.can_attack and enemy.distance_to(b) < 25))
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


def launch(policy, map_name, difficulty, realtime):
    level = {"easy": Difficulty.Easy, "medium": Difficulty.Medium, "hard": Difficulty.Hard}[difficulty]
    run_game(maps.get(map_name), [Bot(Race.Protoss, YuriBot(policy)), Computer(Race.Zerg, level)], realtime=realtime)
