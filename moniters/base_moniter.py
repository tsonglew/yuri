import cv2
from sc2.constants import PROBE


class BaseMoniter:

    def __init__(self, headless):
        self.headless = headless

    async def draw_resources(self, bot, game_data):
        line_max = 50

        (worker_weight, plausible_supply, population_ratio, vespene_ratio, mineral_ratio) \
            = await self.calculate_resources(bot)

        cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)


    async def calculate_resources(self, bot):
        line_max = 50
        mineral_ratio = min(self.minerals / 1500, 1)
        vespene_ratio = min(self.vespene / 1500, 1)
        population_ratio = min(self.supply_left / max(self.supply_cap, 1), 1)
        plausible_supply = self.supply_cap / 200.0
        worker_weight = min(len(self.units(PROBE)) / max(self.supply_cap-self.supply_left, 1), 1)

        return (worker_weight, plausible_supply, population_ratio, vespene_ratio, mineral_ratio)


    async def flip(self, game_data):
        self.flipped = cv2.flip(game_data, 0)

    
    def show(self, title, flipped):
        # dsize = Size(round(fx*src.cols), round(fy*src.rows))
        resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)
        cv2.imshow(str(title), resized)
        cv2.waitKey(1)


    def get_flipped(self):
        return self.flipped
