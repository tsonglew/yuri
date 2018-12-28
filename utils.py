import random
from collections import Iterable


def check_data(no_attacks, attack_closest_to_nexus, 
                attack_enemy_structures, attack_enemy_start):
    """
    1. print num of each kind of attack action
    2. return a list of num of attack actions 
    """
    choices = {
        "no_attacks": no_attacks,
        "attack_closest_to_nexus": attack_closest_to_nexus,
        "attack_enemy_structures": attack_enemy_structures,
        "attack_enemy_start": attack_enemy_start
    }

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        lengths.append(len(choices[choice]))

    print("Total data length now is:", sum(lengths))
    return lengths


def recursive_shuffle(obj, dim=None):
    """
    shuffle object recursively with restricted dimensions
    works the same as random.shuffle when dim=0
    """
    sub_recursive_shuffle(obj, 0, dim)


def sub_recursive_shuffle(obj, cdim=0, dim=None):
    if (dim is not None and cdim > dim) or (not isinstance(obj, Iterable)):
        return
    random.shuffle(obj)
    for i in obj:
        sub_recursive_shuffle(i, cdim + 1, dim)

