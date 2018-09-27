import sc2
from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON

class WorkerBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()

    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

run_game(maps.get('AbyssalReefLE'), [
    Bot(Race.Protoss, WorkerBot()),
    Computer(Race.Terran, Difficulty.Hard)
], realtime=True)