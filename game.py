from mainbot import MainBot

from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer

run_game(maps.get('AbyssalReefLE'), [
    Bot(Race.Protoss, MainBot()),
    Computer(Race.Terran, Difficulty.Hard)
], realtime=False)