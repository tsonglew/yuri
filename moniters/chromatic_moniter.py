from .base_moniter import BaseMoniter

import cv2
import numpy as np
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY


class ChromaticMoniter(BaseMoniter):

    def __init__(self, headless):
        super().__init__(headless)
        self.flipped = None
        self.worker_names = ['probe', 'scv', 'drone']
        self.main_base_names = [
            'nexus', # Protoss
            'hatchery' # Zerg
            'commandcenter', # Terran
            'orbitalcommand',  # Terran(Upgraded)
            'planetaryfortress'  # Terran(Upgraded)
        ]
        self.draw_dict = {
            NEXUS: [15, (0, 255, 0)],
            PYLON: [3, (20, 235, 0)],
            PROBE: [1, (55, 200, 0)],
            ASSIMILATOR: [2, (55, 200, 0)],
            OBSERVER: [1, (255, 255, 255)],
            GATEWAY: [3, (200, 100, 0)],
            CYBERNETICSCORE: [3, (150, 150, 0)],
            ROBOTICSFACILITY: [5, (215, 155, 0)],
            STARGATE: [5, (255, 0, 0)],
            VOIDRAY: [3, (255, 100, 0)]
        }


    async def draw(self, bot):
        """
        convert data into OpenGL images
        """
        game_data = np.zeros(
            (bot.game_info.map_size[1], bot.game_info.map_size[0], 3),
            dtype=np.uint8
        )
        await self.draw_own_units(bot, game_data)
        await self.draw_enemy_buildings(bot, game_data)
        await self.draw_enemy_units(bot, game_data)
        await self.draw_resources(bot, game_data)
        await self.flip(game_data)

        if not self.headless:
            self.show(bot.title, self.flipped)


    async def draw_own_units(self, bot, game_data):
        """
        draw bot own units
        """
        for unit_type in self.draw_dict:
            for unit in bot.units(unit_type):
                pos = unit.position
                cv2.circle(
                    game_data,
                    (int(pos[0]), int(pos[1])),
                    self.draw_dict[unit_type][0],
                    self.draw_dict[unit_type][1],
                    -1
                )


    async def draw_enemy_buildings(self, bot, game_data):
        for enemy_building in bot.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in self.main_base_names:
                await self.draw_enemy_main_base(game_data, pos)
            else:
                await self.draw_anonymous_enemy_building(game_data, pos)
        

    async def draw_enemy_main_base(self, game_data, pos):
        cv2.circle(
            game_data,
            (int(pos[0]), int(pos[1])),
            15,
            (0, 0, 255),
            -1
        )


    async def draw_anonymous_enemy_building(self, game_data, pos):
        cv2.circle(
            game_data,
            (int(pos[0]), int(pos[1])),
            5,
            (200, 50, 212),
            -1
        )


    async def draw_enemy_units(self, bot, game_data):
        for enemy_unit in bot.known_enemy_units:
            if not enemy_unit.is_structure:
                pos = enemy_unit.position
                if enemy_unit.name.lower() in self.worker_names:
                    await self.draw_enemy_worker(game_data, pos)
                else:
                    await self.draw_anonymous_enemy_units(game_data, pos)


    async def draw_enemy_worker(self, game_data, pos):
        cv2.circle(
            game_data, 
            (int(pos[0]), int(pos[1])), 
            1, 
            (55, 0, 155), 
            -1
        )
    
    async def draw_anonymous_enemy_units(self, game_data, pos):
        cv2.circle(
            game_data, 
            (int(pos[0]), int(pos[1])), 
            3, 
            (50, 0, 215), 
            -1
        )
