from .attackbot import AttackBot
from .buildbot import BuildBot
from .expandbot import ExpandBot
from .scoutbot import ScoutBot
from .databot import DataBot
from ..loggers import logger

import numpy as np

import random


class FullChoiceBot(AttackBot, BuildBot, ExpandBot, ScoutBot, DataBot):

    def __init__(self):
        AttackBot.__init__(self)
        BuildBot.__init__(self)
        ExpandBot.__init__(self)
        ScoutBot.__init__(self)
        DataBot.__init__(self)

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
        await self.monitor.draw(self)
        await self.do_something()

    async def do_something(self):
        if self.minute > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.monitor.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 14)

            (func, name) = self.choices[choice]

            logger.debug(name)
            await func()

            choice_array = np.zeros(14)
            choice_array[choice] = 1
            new_data = [choice_array, self.monitor.flipped]
            self.train_data.append(new_data)
            return new_data

    async def do_nothing(self):
        wait = random.randrange(7, 100) / 100
        self.do_something_after = self.minute + wait


class AttackChoiceBot(AttackBot, BuildBot, ExpandBot, DataBot, ScoutBot):

    def __init__(self):
        AttackBot.__init__(self)
        BuildBot.__init__(self)
        ExpandBot.__init__(self)
        DataBot.__init__(self)
        ScoutBot.__init__(self)

    async def on_step(self, iteration):
        """
        actions will be made every step 
        """
        self.iteration = iteration
        self.minute = self.time / 60
        await self.distribute_workers()
        await self.monitor.draw(self)
        await self.build_worker()
        await self.build_pylon()
        await self.build_assimilator()
        await self.expand()
        await self.build_offensive_force_building()
        await self.build_offensive_force()
        await self.scout()

        return await self.attack(self.monitor, self.use_model)
