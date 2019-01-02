import cv2
import numpy as np
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY

class CV2Moniter:
    def __init__(self, headless):
        self.headless = headless
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
            # dsize = Size(round(fx*src.cols), round(fy*src.rows))
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Moniter', resized)
            cv2.waitKey(1)

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

    async def draw_resources(self, bot, game_data):
        """
        draw mineral, vespene lines
        """
        line_max = 50

        military_ratio, plausible_supply, population_ratio, vespene_ratio, \
            mineral_ratio = await self.calculate_resources(bot)
        cv2.line(game_data, (0, 19), (int(line_max*military_ratio), 19), (250, 250, 200), 3) # worker & supply
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)

    async def calculate_resources(self, bot):
        mineral_ratio = min([bot.minerals / 1500, 1.0])
        vespene_ratio = min([bot.vespene / 1500, 1.0])
        population_ratio = min([bot.supply_left / bot.supply_cap, 1.0])
        plausible_supply = bot.supply_cap / 200 # Total available supply / 200(Max limited population)
        military_ratio = min([
            (len(bot.units(VOIDRAY))+len(bot.units(STALKER))) /
            max(bot.supply_cap - bot.supply_left, 1.0),
            1.0
        ])
        return military_ratio, plausible_supply, population_ratio, \
                vespene_ratio, mineral_ratio
    
    async def flip(self, game_data):
        self.flipped = cv2.flip(game_data, 0)

    def get_flipped(self):
        return self.flipped
