import random

from ..loggers import logger
import sc2
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, ROBOTICSFACILITY, \
    OBSERVER, ZEALOT


class BuildBot(sc2.BotAI):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build_pylon_supply_left = 5
        self.nexus_max = 3


    async def build_worker(self):
        logger.debug('called build_worker')
        probe_nums = len(self.units(PROBE))
        if len(self.units(NEXUS)) * 16 > probe_nums and probe_nums < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))


    async def build_pylon(self):
        """
        build pylons near nexuses if needed
        """
        logger.debug('called build_pylon')
        if self.supply_left < self.build_pylon_supply_left and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)


    async def build_assimilator(self):
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


    async def offensive_force_building(self):
        """
        build gateway and stargate
        """
        await random.choice([self.build_gateway, self.build_stargate])()


    async def build_gateway(self):
        if not self.units(PYLON).ready.exists:
            return
        pylon = self.units(PYLON).ready.random

        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)
        elif self.units(GATEWAY).amount == 0:
            if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                await self.build(GATEWAY, near=pylon)


    async def build_stargate(self):
        if not self.units(PYLON).ready.exists:
            return
        pylon = self.units(PYLON).ready.random
        
        if self.units(CYBERNETICSCORE).ready.exists:
            if len(self.units(STARGATE)) < self.minute:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)
            if self.units(ROBOTICSFACILITY).amount == 0:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)


    async def build_offensive_force(self):
        """
        train stalker | zealot | voidray
        """
        await random.choice([
            self.build_stalker, 
            self.build_voidray,
            self.build_zealot
        ])()


    async def build_stalker(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(STALKER).amount <= self.units(VOIDRAY).amount \
                    and self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))
    

    async def build_voidray(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))


    async def build_scout(self):
        """
        train observers
        """
        if len(self.units(OBSERVER)) < self.minute / 3:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))


    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))
