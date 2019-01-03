from .loggers import logger

import random

from sc2 import run_game, Race, maps, Difficulty, Result
from sc2.player import Bot, Computer


class GameLauncher:

    def __init__(self, bot, use_model, model_path, realtime):
        logger.debug('Game Launcher inited')
        self.map = 'AbyssalReefLE'
        self.bot = bot
        self.use_model = use_model
        self.model_path = model_path
        self.realtime = realtime


    def create_bot(self):
        return Bot(
            Race.Protoss, 
            self.bot(self.use_model, self.model_path)
        )

    
    def create_computer(self):
        race = random.choice([Race.Terran, Race.Zerg, Race.Protoss])
        difficulty = Difficulty.Easy
        logger.debug(f'To compete: {race} {difficulty}')
        return Computer(race, difficulty)


    def start_game(self):
        result = run_game(maps.get(self.map), [
            self.create_bot(),
            self.create_computer()
        ], realtime=self.realtime)
        return result


    def get_train_data(self):
        return self.bot.train_data