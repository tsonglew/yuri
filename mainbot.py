from .basebots import *
from .utils import *
from .moniters import ChromaticMoniter, MonochromeMoniter
from .loggers import logger

import random

import keras
import numpy as np
import sc2
from sc2 import position


class MainBot(ScoutBot, AttackBot, BuildBot, ExpandBot):
    def __init__(self, use_model, name, model_path=None):
        super().__init__()
        self.choices = [
            (self.build_scout, 'build scout'),
            (self.build_zealot, 'build zealot'),
            (self.build_gateway, 'build gateway'),
            (self.build_voidray, 'build voidray'),
            (self.build_stalker, 'build stalker'),
            (self.build_worker, 'build worker'),
            (self.build_assimilator, 'build assimilator'),
            (self.build_stargate, 'build stargate'),
            (self.build_pylon, 'build pylon'),
            (self.defend_nexus, 'defend nexus'),
            (self.attack_known_enemy_unit, 'attack known enemy unit'),
            (self.attack_known_enemy_structure, 'attack known enemy structure'),
            (self.expand, 'expand'),
            (self.do_nothing, 'do_nothing')
        ]
        self.IPS = 165  # probable Iteration Per Second
        self.MAX_WORKERS = 65
        self.use_model = use_model
        self.train_data = list()
        self.moniter = MonochromeMoniter(headless=False)

        if self.use_model:
            logger.info(f'Running game with BaiscCNN model: {model_path}')
            self.model = keras.models.load_model(model_path)

        logger.debug(f'inited bot')
  

    async def on_step(self, iteration):
        """
        actions will be made every step 
        """
        # self.minute defined in Dentosal/python-sc2/sc2/bot_ai/BotAI
        # self.minute = self.state.game_loop / 22.4 # / (1/1.4) * (1/16)
        # approximately count in the second in game
        self.minute = self.time / 60

        await self.distribute_workers()
        await self.scout()
        await self.moniter.draw(self)
        await self.do_something()


    async def do_something(self):
        if self.minute > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.moniter.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 14)

            (func, name) = self.choices[choice]

            logger.debug(name)
            await func()

            choice_array = np.zeros(14)
            choice_array[choice] = 1
            self.train_data.append([choice_array, self.moniter.flipped])


    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.minute + wait


    def find_target(self):
        """
        choose random known enemy units and structures. If none of them is uknown,
        reutrn the location where enemy starts
        """
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        if len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        return self.enemy_start_locations[0]
