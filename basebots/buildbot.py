import random

from ..loggers import logger
import sc2
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, ROBOTICSFACILITY, \
    OBSERVER, ZEALOT


class BuildBot(sc2.BotAI):

    def __init__(self, *args, **kwargs):
        self.build_pylon_supply_left = 5
        self.nexus_max = 3
        self.workers_max = 65

    async def build_worker(self):
        probe_nums = len(self.units(PROBE))
        if len(self.units(NEXUS)) * 16 > probe_nums and probe_nums < self.workers_max:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    logger.debug('called build_worker')
                    await self.do(nexus.train(PROBE))

    async def build_pylon(self):
        """
        build pylons near nexuses if needed
        """
        if self.supply_left < self.build_pylon_supply_left and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    logger.debug('called build_pylon')
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilator(self):
        """
        build assimilators on vespene geyser near nexuses
        """
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    return
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    return
                if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    logger.debug('build assimilator')
                    await self.do(worker.build(ASSIMILATOR, vespene))

    async def build_offensive_force_building(self):
        # chose random pylon
        if not self.units(PYLON).ready.exists:
            return
        pylon = self.units(PYLON).ready.random

        if self.units(GATEWAY).ready.exists and self.units(CYBERNETICSCORE).amount == 0:
            if self.can_afford(CYBERNETICSCORE):
                if not self.already_pending(CYBERNETICSCORE):
                    logger.debug('build cyberneticscore')
                    await self.build(CYBERNETICSCORE, near=pylon)
                # else:
            #         logger.debug('cyberneticscore already pending')
            # else:
            #     logger.debug('cannot afford cyberneticscore')
        elif self.units(GATEWAY).amount == 0:
            if self.can_afford(GATEWAY):
                if not self.already_pending(GATEWAY):
                    logger.debug('build gateway')
                    await self.build(GATEWAY, near=pylon)
            #     else:
            #         logger.debug('gateway already pending')
            # else:
            #     logger.debug('cannot afford gateway')

        if self.units(CYBERNETICSCORE).ready.exists:
            stargate_num = self.units(STARGATE).amount
            if stargate_num < self.time and stargate_num < 4:
                if self.can_afford(STARGATE):
                    logger.debug('build stargate')
                    await self.build(STARGATE, near=pylon)
        #         else:
        #             logger.debug('cannot afford stargate')
        #     else:
        #         logger.debug('not time to build stargate')
        # else:
        #     logger.debug('cyberneticscore not ready')

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
        # train stalkers
        if self.units(STARGATE).ready.amount < 2:
            for gw in self.units(GATEWAY).ready.noqueue:
                if self.can_afford(STALKER) and self.supply_left > 0:
                    await self.do(gw.train(STALKER))

        # train voidrays
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    async def build_stalker(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(STALKER).amount <= self.units(VOIDRAY).amount \
                    and self.can_afford(STALKER) and self.supply_left > 0:
                logger.debug('train stalker')
                await self.do(gw.train(STALKER))

    async def build_voidray(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                logger.debug('train voidray')
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
