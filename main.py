import random

import sc2
from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER


BUILD_PYLON_SUPPLY_LEFT = 5
NEXUS_MAX = 3
OFFENCE_AMOUNT = 15
DEFENCE_AMOUNT = 3

class MainBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_building()
        await self.build_offensive_force()
        await self.attack()

    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < BUILD_PYLON_SUPPLY_LEFT and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))

    async def expand(self):
        if self.units(NEXUS).amount < NEXUS_MAX and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_building(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists:
                if not self.units(CYBERNETICSCORE):
                    if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                        await self.build(CYBERNETICSCORE, near=pylon)
            else:
               if self.can_afford(GATEWAY) and self.units(GATEWAY).amount < 4:
                   await self.build(GATEWAY, near=pylon)

    async def build_offensive_force(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))

    def find_target(self):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        if len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        return self.enemy_start_locations[0]

    async def attack(self):
        if self.units(STALKER).amount > OFFENCE_AMOUNT:
            for s in self.units(STALKER).idle:
                await self.do(s.attack(self.find_target()))
        elif self.units(STALKER).amount > DEFENCE_AMOUNT:
            if len(self.known_enemy_units) > 0:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))

run_game(maps.get('AbyssalReefLE'), [
    Bot(Race.Protoss, MainBot()),
    Computer(Race.Terran, Difficulty.Medium)
], realtime=True)