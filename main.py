import sc2
from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer

class WorkerBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()

run_game(maps.get('AbyssalReefLE'), [
    Bot(Race.Protoss, WorkerBot()),
    Computer(Race.Terran, Difficulty.Hard)
], realtime=True)