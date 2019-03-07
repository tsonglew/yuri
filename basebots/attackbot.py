from .waitbot import WaitBot

import random

from ..loggers import logger
import numpy as np
import sc2
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, ZEALOT, ROBOTICSFACILITY


class AttackBot(sc2.BotAI, WaitBot):

    def __init__(self, *args, **kwargs):
        WaitBot.__init__(self)

        self.choice_funcs = [
            (self.no_attack, 'no_attack'),
            (self.defend_nexus, 'defend nexus'),
            (self.attack_known_enemy_structure, 'attack enemy structure'),
            (self.attack_enemy_start, 'attack enemy start'),
            (self.attack_known_enemy_unit, 'attack known enemy unit')
        ]

    async def attack(self, monitor, use_model) -> [np.array, np.array]:
        """
        voidrays choose random enemy units or structures to attack; used for
        collecting training data
        """
        if len(self.units(VOIDRAY).idle) > 0:

            if self.minute > self.do_something_after:
                flipped = monitor.get_flipped()
                if use_model:
                    choice = await self.predict_attack_choice(flipped)
                else:
                    choice = random.randrange(0, 4)

                target = await self.choice_funcs[choice][0]()
                await self.attack_unit(target)

                choice_array = np.zeros(4)
                choice_array[choice] = 1
                # Training data consits of two tensors, which are random choice
                # array(1*4) and game_data map(176*200*3)
                return [choice_array, flipped]
            return None

    async def predict_attack_choice(self, flipped) -> int:
        prediction = self.model.predict([flipped.reshape([-1, 176, 200, 3])])
        choice = np.argmax(prediction[0])
        logger.debug(f'Attack Choice #{choice}:{self.choice_funcs[choice][1]}')
        return choice

    async def no_attack(self) -> None:
        wait = random.randrange(10, 50) / 100
        self.do_something_after = self.minute + wait
        return None

    async def defend_nexus(self) -> sc2.unit:
        if len(self.known_enemy_units) > 0:
            own_nexuses = self.units(NEXUS)
            if len(own_nexuses) > 0:
                target = self.known_enemy_units.closest_to(random.choice(own_nexuses))
                await self.attack_unit(target)

    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            target = random.choice(self.known_enemy_units)
            await self.attack_unit(target)

    async def attack_known_enemy_structure(self):
        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures)
            await self.attack_unit(target)

    async def attack_enemy_start(self):
        target = self.enemy_start_locations[0]
        await self.attack_unit(target)

    async def attack_unit(self, target):
        if target is not None:
            for offensive_type in (STALKER, VOIDRAY, ZEALOT):
                for unit in self.units(offensive_type):
                    await self.do(unit.attack(target))
