from ..loggers import logger

import sc2
from sc2.constants import NEXUS


class ExpandBot:

    def __init__(self, *args, **kwargs):
        super().__init__()

    async def expand(self):
        """
        build more nuxuses if affordable
        """
        try:
            if self.units(NEXUS).amount < self.minute and self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            logger.error(e)
