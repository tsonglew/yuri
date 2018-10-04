import os
import random

import sc2
from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY


BUILD_PYLON_SUPPLY_LEFT = 5
NEXUS_MAX = 3
OFFENCE_AMOUNT = 15
DEFENCE_AMOUNT = 3

class MainBot(sc2.BotAI):
    def __init__(self):
        self.IPS = 165  # Iteration Per Second
        self.MAX_WORKERS = 65

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_building()
        await self.build_offensive_force()
        await self.attack()

    async def build_workers(self):
        probe_nums = len(self.units(PROBE))
        if len(self.units(NEXUS)) * 16 > probe_nums and probe_nums < self.MAX_WORKERS:
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
        if not self.units(PYLON).ready.exists:
            return
        pylon = self.units(PYLON).ready.random
        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)
        elif len(self.units(GATEWAY)) < self.iteration / self.IPS / 2:
            if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                await self.build(GATEWAY, near=pylon)
        
        if self.units(CYBERNETICSCORE).ready.exists:
            if len(self.units(STARGATE)) < self.iteration / self.IPS / 2:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        # train stalkers
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(STALKER).amount < self.units(VOIDRAY).amount \
                    and self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))

        # train void-rays
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    def find_target(self):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        if len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        return self.enemy_start_locations[0]

    async def offend(self, unit):
        for s in self.units(unit).idle:
            await self.do(s.attack(random.choice(self.known_enemy_units)))
    
    async def defend(self, unit):
        for s in self.units(unit).idle:
            await self.do(s.attack(self.find_target()))

    async def attack(self):
        aggresive_units = {
            STALKER: [15, 3],
            VOIDRAY: [8, 3]
        }
        for unit in aggresive_units:
            if self.units(unit).amount > aggresive_units[unit][0]:
                await self.offend(unit)
            elif self.units(unit).amount > aggresive_units[unit][1]:
                await self.defend(unit)


if __name__ == '__main__':
    run_game(maps.get('AbyssalReefLE'), [
        Bot(Race.Protoss, MainBot()),
        Computer(Race.Terran, Difficulty.Hard)
    ], realtime=False)

    os.system('defaults write -g com.apple.keyboard.fnState -boolean false')
