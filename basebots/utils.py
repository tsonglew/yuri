import random

from sc2 import position


def random_location_variance(location, x_max, y_max) -> position.Point2:
    """
    return a random position near enemy start location; used for scouting
    """
    x = location[0] + random.randrange(-5, 5)
    y = location[1] + random.randrange(-5, 5)

    # make the out-of-map positions valid 
    x = max([x, 0])
    y = max([y, 0])
    x = min([x, x_max])
    y = min([y, y_max])
    return position.Point2(position.Pointlike((x, y)))
