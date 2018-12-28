import os
import random
from collections import Iterable

import numpy as np


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


def load_attack_train_data(train_data_dir, files, test_size):
    """
    Load attack train data from file to np.array and shuffle
    """
    no_attacks, attack_closest_to_nexus, attack_enemy_structures, \
        attack_enemy_start = load_attack_data_file(train_data_dir, files)
    
    lengths = check_data(no_attacks, attack_closest_to_nexus,
                        attack_enemy_structures, attack_enemy_start)
    lowest_length = min(lengths)
    recursive_shuffle([no_attacks, attack_closest_to_nexus, 
                        attack_enemy_structures, attack_enemy_start], 1)
    no_attacks = no_attacks[:lowest_length]
    attack_closest_to_nexus = attack_closest_to_nexus[:lowest_length]
    attack_enemy_structures = attack_enemy_structures[:lowest_length]
    attack_enemy_start = attack_enemy_start[:lowest_length]
    check_data(no_attacks, attack_closest_to_nexus,
                attack_enemy_structures, attack_enemy_start)

    train_data = no_attacks + attack_closest_to_nexus \
                    + attack_enemy_start + attack_enemy_structures
    return split_shuffle_train_test_data(train_data, test_size)


def load_attack_data_file(train_data_dir, files):
    no_attacks = []
    attack_closest_to_nexus = []
    attack_enemy_structures = []
    attack_enemy_start = []

    for file in files:
        full_path = os.path.join(train_data_dir, file)
        data = np.load(full_path)
        data = list(data)

        # classify map data by attack action
        for d in data:
            choice = np.argmax(d[0])
            if choice == 0:
                no_attacks.append([d[0], d[1]])
            elif choice == 1:
                attack_closest_to_nexus.append([d[0], d[1]])
            elif choice == 2:
                attack_enemy_structures.append([d[0], d[1]])
            elif choice == 3:
                attack_enemy_start.append([d[0], d[1]])
    return no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start


def split_shuffle_train_test_data(train_data, test_size):
    random.shuffle(train_data)

    x_train = np.array([i[1] for i in train_data[:-test_size]])
    print(f'x_train original shape: {x_train.shape}')
    x_train = x_train.reshape(-1, 176, 200, 3)
    print(f'x_train shape after reshaping: {x_train.shape}')
    y_train = np.array([i[0] for i in train_data[:-test_size]])

    x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
    y_test = np.array([i[0] for i in train_data[-test_size:]])
    return x_train, y_train, x_test, y_test


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
