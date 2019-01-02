from .utils import random_location_variance

import math

import sc2
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY


# Scout quantity limits
OBSERVER_SCOUT_MAX_NUM = 3
PROBE_SCOUT_MAX_NUM = 1


class ScoutBot(sc2.BotAI):
    def __init__(self):
        # sount and expand
        self.expand_distance_location = dict()
        self.scouts_and_spots = dict()
        self.expand_dis = dict()

    async def scout(self):
        """
        scout with probes or observers
        """
        await self.calculate_epansion_distances()
        await self.update_scouts_and_spots()
        (unit_type, assign, scouting_probe) = await self.get_unit_type_and_num_limit()

        # No idle obs or already assigned probe
        idle_candidates = self.units(unit_type).idle
        if not assign and scouting_probe is not None:
            scout_target = random_location_variance(
                self.scouts_and_spots[scouting_probe.tag],
                self.game_info.map_size[0],
                self.game_info.map_size[1]
            )
            print(f'Go to scout {scout_target}')
            await self.do(scouting_probe.move(scout_target))
            return

        if len(idle_candidates) <= 0:
            return

        if unit_type == PROBE:
            await self.ob_scout(idle_candidates[:PROBE_SCOUT_MAX_NUM])
        elif unit_type == OBSERVER:
            await self.ob_scout(idle_candidates[:OBSERVER_SCOUT_MAX_NUM])

    
    async def calculate_epansion_distances(self):
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_distance_location[distance_to_enemy_start] = el

        self.expand_location_distance = {v: k for k, v in self.expand_distance_location.items()}
        self.ordered_expand_distances = sorted(k for k in self.expand_distance_location.keys())


    async def update_scouts_and_spots(self):
        """
        remove dead from scouts and spots dict
        """
        existing_ids = set([unit.tag for unit in self.units])
        scouts_and_spots_keys = list(self.scouts_and_spots.keys())
        for scout in scouts_and_spots_keys:
            if scout not in existing_ids:
                del self.scouts_and_spots[scout]


    async def get_unit_type_and_num_limit(self) -> (sc2.constants, bool, sc2.unit):
        assign_scout = True
        scouting_probe = None
        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            # stop assigning if assigned
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots.keys():
                    assign_scout = False
                    scouting_probe = unit
        else:
            unit_type = OBSERVER

        return (unit_type, assign_scout, scouting_probe)


    async def ob_scout(self, obs):
        for ob in obs:
            if ob.tag not in self.scouts_and_spots:
                for dist in self.ordered_expand_distances:
                    location = self.expand_distance_location.get(dist)
                    if location is not None and \
                            location not in self.scouts_and_spots.values():
                        await self.do(ob.move(location))
                        self.scouts_and_spots[ob.tag] = location
                        break


    async def build_scout(self):
        """
        train observers
        """
        if len(self.units(OBSERVER)) < self.time//(60*3):
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))
