from .loggers import logger

import random

from sc2 import run_game, Race, maps, Difficulty
from sc2.player import Bot, Computer


class GameLauncher:

    def __init__(self, bot, use_model, model_path, realtime):
        logger.debug('Game Launcher inited')
        self.map = 'AbyssalReefLE'
        self.bot = bot
        self.use_model = use_model
        self.model_path = model_path
        self.realtime = realtime
        self.difficulty_dict = {
            'easy': Difficulty.Easy,
            'medium': Difficulty.Medium,
            'hard': Difficulty.Hard
        }
        self.race_dict = {
            'zerg': Race.Zerg,
            'terran': Race.Terran,
            'Protoss': Race.Protoss
        }

    def create_bot(self, bot_title, train_data_tensor):
        return Bot(
            Race.Protoss,
            self.bot(train_data_tensor, self.use_model, bot_title, self.model_path)
        )

    def start_game(self, difficulty, race, train_data_tensor):
        result = run_game(maps.get(self.map), [
            self.create_bot('bot 1', train_data_tensor),
            self.create_computer(difficulty, race)
        ], realtime=self.realtime)
        return result

    def get_train_data(self):
        return self.bot.get_data()

    def create_computer(self, d, r):
        logger.debug(f'To compete: {self.race_dict[r]} {self.difficulty_dict[d]}')
        return Computer(self.race_dict[r], self.difficulty_dict[d])
