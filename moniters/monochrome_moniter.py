from .base_moniter import BaseMoniter

import math

import cv2
import numpy as np


class MonochromeMoniter(BaseMoniter):

    def __init__(self, headless):
        super().__init__(headless)
        self.ally_color = (255, 255, 255)
        self.enemy_color = (125, 125, 125)


    async def draw_grey(self, bot):
        game_data = np.zeros(
            (bot.game_info.map_size[1], bot.game_info.map_size[0], 3),
            np.uint8
        )
        await self.draw_ally(bot, game_data)
        await self.draw_enemy(bot, game_data)
        await self.draw_resources(bot, game_data)

        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flip(grayed)

        if not self.headless:
            self.show(bot.title, self.fipped)


    async def draw_ally(self, bot, game_data):
        for unit in bot.units().ready:
            pos = unit.position
            cv2.circle(
                img=game_data,
                center=(int(pos[0]), int(pos[1])),
                radius=int(unit.radius*8), 
                color=self.ally_color,
                thickness=math.ceil(int(unit.radius*0.5))
            )


    async def draw_enemy(self, bot, game_data):
        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(
                img=game_data, 
                center=(int(pos[0]), int(pos[1])), 
                radius=int(unit.radius*8), 
                color=self.enemy_color,
                thickness=math.ceil(int(unit.radius*0.5))
            )
