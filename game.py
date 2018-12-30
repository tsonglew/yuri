from mainbot import MainBot

import argparse
from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer

parser = argparse.ArgumentParser(
    prog='game.py',
    usage='python3 %(prog) [--use_model]',
    description='Run game'
)
parser.add_argument(
    '--use_model', action='store_true',
    help='use a trained model'
)

cmd_args = parser.parse_args()

run_game(maps.get('AbyssalReefLE'), [
    Bot(Race.Protoss, MainBot(cmd_args.use_model)),
    Computer(Race.Terran, Difficulty.Hard)
], realtime=False)