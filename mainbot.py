from basebots import *
from utils import *
from moniter import CV2Moniter

import random

import keras
import numpy as np
import sc2
from sc2 import position
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY


HEADLESS = False
BUILD_PYLON_SUPPLY_LEFT = 5
NEXUS_MAX = 3
OFFENCE_AMOUNT = 15
DEFENCE_AMOUNT = 3


class MainBot(ScoutBot):
    def __init__(self, use_model):
        super().__init__()
        self.IPS = 165  # probable Iteration Per Second
        self.MAX_WORKERS = 65
        self.do_something_after = 0
        self.use_model = use_model
        self.train_data = list()
        self.moniter = CV2Moniter(HEADLESS)

        if self.use_model:
            print('Running game with BaiscCNN model')
            model_path = 'BasicCNN-10-epochs-0.0001-LR-STAGE1'
            self.model = keras.models.load_model(model_path)


    async def on_step(self, iteration):
        """
        actions will be made every step 
        """
        # self.minute defined in Dentosal/python-sc2/sc2/bot_ai/BotAI
        # self.minute = self.state.game_loop / 22.4 # / (1/1.4) * (1/16)
        # approximately count in the second in game
        self.minute = self.time / 60
        print(f'minute: {self.minute}')

        await self.build_scout()
        await self.scout()
        await self.moniter.draw(self)
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_building()
        await self.build_offensive_force()
        await self.attack()


    def on_end(self, game_result):
        """
        save training data if win
        """
        print('--- on_end called ---')
        print(f'Result: {game_result}')
        print(f'Using model: {self.use_model}')

        if game_result == Result.Victory:
            np.save("local_train/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

        with open('log.txt', 'a') as f:
            prefix = 'Model: ' if self.use_model else 'Random: '
            f.write(f'{prefix}{game_result}\n')



    async def build_workers(self):
        probe_nums = len(self.units(PROBE))
        if len(self.units(NEXUS)) * 16 > probe_nums and probe_nums < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))


    async def build_pylons(self):
        """
        build pylons near nexuses if needed
        """
        if self.supply_left < BUILD_PYLON_SUPPLY_LEFT and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)


    async def build_assimilators(self):
        """
        build assimilators on vespene geyser near nexuses
        """
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(ASSIMILATOR, vespene))


    async def expand(self):
        """
        build more nuxuses if affordable
        """
        try:
            if self.units(NEXUS).amount < self.minute/2 and \
                    self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(e)


    async def offensive_force_building(self):
        if not self.units(PYLON).ready.exists:
            return
        pylon = self.units(PYLON).ready.random
        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)
        elif self.units(GATEWAY).amount == 0:
            if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                await self.build(GATEWAY, near=pylon)
        
        if self.units(CYBERNETICSCORE).ready.exists:
            if len(self.units(STARGATE)) < self.minute:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)
            if self.units(ROBOTICSFACILITY).amount == 0:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)


    async def build_offensive_force(self):
        # train stalkers
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(STALKER).amount <= self.units(VOIDRAY).amount \
                    and self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))


        # train void-rays
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))


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


    async def offend(self, units):
        if not isinstance(units, list):
            units = [units]
        for u in units:
            for s in self.units(unit).idle:
                await self.do(s.attack(random.choice(self.known_enemy_units)))

    
    async def defend(self, units):
        if not isinstance(units, list):
            units = [units]
        for u in units:
            for s in self.units(u).idle:
                await self.do(s.attack(self.find_target()))


    async def attack(self):
        """
        voidrays choose random enemy units or structures to attack; used for 
        collecting training data
        """
        if len(self.units(VOIDRAY).idle) > 0:
            target = False
            if self.minute > self.do_something_after:
                if self.use_model:
                    flipped = self.moniter.get_flipped()
                    prediction = self.model.predict([flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])

                    choice_dict = {
                        0: "No Attack!",
                        1: "Attack close to our nexus!",
                        2: "Attack Enemy Structure!",
                        3: "Attack Eneemy Start!"
                    }
                    print("Choice #{}:{}".format(choice, choice_dict[choice]))
                else:
                    choice = random.randrange(0, 4)

                if choice == 0:
                    # no attack
                    wait = random.randrange(7,100)/100
                    self.do_something_after = self.time + wait

                elif choice == 1:
                    # attack unit closest nexus
                    if len(self.known_enemy_units) > 0:
                        own_nexuses = self.units(NEXUS)
                        if len(own_nexuses) > 0:
                            target = self.known_enemy_units.closest_to(random.choice(own_nexuses))

                elif choice == 2:
                    # attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    # attack enemy start
                    target = self.enemy_start_locations[0]

                if target:
                    for offensive_type in (STALKER, VOIDRAY):
                        for unit in self.units(offensive_type).idle:
                            await self.do(unit.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                print(y)
                # Training data consits of two tensors, which are random choice
                # array(1*4) and game_data map(176*200*3)
                self.train_data.append([y,self.moniter.flipped])
    