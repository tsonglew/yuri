from ..loggers import logger

import sc2
from sc2.constants import NEXUS


class ExpandBot:

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.nexus_max = 7

    async def expand(self):
        """
        build more nuxuses if affordable
        """
        try:
            if self.units(NEXUS).amount < self.minute and self.units(NEXUS).amount < self.nexus_max and self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            logger.error(e)
