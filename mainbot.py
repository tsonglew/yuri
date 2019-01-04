from .basebots import AttackChoiceBot, FullChoiceBot
from .loggers import logger
from .monitors import MonochromeMonitor, ChromaticMonitor

import random

import keras


class MainBot(AttackChoiceBot):

    def __init__(self, train_data_tensor, use_model, title, model_path=None):
        if isinstance(self, AttackChoiceBot):
            AttackChoiceBot.__init__(self)
            monitor_class = ChromaticMonitor
        elif isinstance(self, FullChoiceBot):
            FullChoiceBot.__init__(self)
            monitor_class = MonochromeMonitor

        self.title = title
        self.IPS = 165  # probable Iteration Per Second
        self.use_model = use_model
        self.monitor = monitor_class(headless=False)
        self.train_data_tensor = train_data_tensor

        if self.use_model:
            logger.info(f'Running game with model: {model_path}')
            self.model = keras.models.load_model(model_path)
        logger.debug(f'inited bot')

    def find_target(self):
        """
        choose random known enemy units and structures. If none of them is unknown,
        return the location where enemy starts
        """
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        if len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        return self.enemy_start_locations[0]

    async def on_step(self, iteration):
        if isinstance(self, AttackChoiceBot):
            new_data = await AttackChoiceBot.on_step(self, iteration)
        elif isinstance(self, FullChoiceBot):
            new_data = await FullChoiceBot.on_step(self, iteration)
        self.train_data_tensor.append(new_data)
