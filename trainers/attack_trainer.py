from .base_trainer import BaseTrainer
from ..models import AttackCNNModel
from ..loggers import logger

import os
import random
from collections import Iterable

import numpy as np


class AttackTrainer(BaseTrainer):

    def __init__(self, config_json):
        BaseTrainer.__init__(self, config_json)
        self.name = 'attackTrainer'
        self.model = AttackCNNModel()

    def train(self):
        for i in range(self.hm_epochs):
            current = 0

            all_files = os.listdir(self.train_data_dir)
            total_files_num = len(all_files)
            logger.info(f'Training file num: {total_files_num}')
            random.shuffle(all_files)

            while current <= total_files_num:
                logger.info(f"Model {id(self.model)} currently doing {current}:{current + self.increment}")

                # Read 200 files per hm_epoch
                x_train, y_train, x_test, y_test = self.load_attack_train_data(
                    self.train_data_dir,
                    all_files[current:current + self.increment],
                    self.test_size
                )
                self.fit(x_train, y_train, x_test, y_test)
                save_path = os.path.join(
                    os.path.dirname(__file__),
                    f'AttackTrainer-{self.hm_epochs}-epochs-{self.learning_rate}'
                )
                self.model.save(save_path)
                current += self.increment

    @staticmethod
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
            lengths.append(len(choices[choice]))

        logger.debug(f'Checking data length: {sum(lengths)}')
        return lengths

    def load_attack_train_data(self, train_data_dir, files, test_size):
        """
        Load attack train data from file to np.array and shuffle
        """
        no_attacks, attack_closest_to_nexus, attack_enemy_structures, \
            attack_enemy_start = self.load_attack_data_file(train_data_dir, files)

        lengths = self.check_data(no_attacks, attack_closest_to_nexus,
                                  attack_enemy_structures, attack_enemy_start)
        lowest_length = min(lengths)
        self.recursive_shuffle([no_attacks, attack_closest_to_nexus,
                                attack_enemy_structures, attack_enemy_start], 1)
        no_attacks = no_attacks[:lowest_length]
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_length]
        attack_enemy_structures = attack_enemy_structures[:lowest_length]
        attack_enemy_start = attack_enemy_start[:lowest_length]
        self.check_data(no_attacks, attack_closest_to_nexus,
                        attack_enemy_structures, attack_enemy_start)

        train_data = no_attacks + attack_closest_to_nexus \
            + attack_enemy_start + attack_enemy_structures
        return self.split_shuffle_train_test_data(train_data, test_size)

    @staticmethod
    def load_attack_data_file(train_data_dir, files):
        """
        load training data for attack macro actions
        """
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

    @staticmethod
    def split_shuffle_train_test_data(train_data, test_size):
        """
        split training data into train data and test data then shuffle
        """
        random.shuffle(train_data)

        x_train = np.array([i[1] for i in train_data[:-test_size]])
        x_train = x_train.reshape(-1, 176, 200, 3)
        y_train = np.array([i[0] for i in train_data[:-test_size]])

        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
        y_test = np.array([i[0] for i in train_data[-test_size:]])
        return x_train, y_train, x_test, y_test

    def recursive_shuffle(self, obj, dim=None):
        """
        shuffle object recursively with restricted dimensions
        works the same as random.shuffle when dim=0
        """
        self.sub_recursive_shuffle(obj, 0, dim)

    def sub_recursive_shuffle(self, obj, cdim=0, dim=None):
        if (dim is not None and cdim > dim) or (not isinstance(obj, Iterable)):
            return
        random.shuffle(obj)
        for i in obj:
            self.sub_recursive_shuffle(i, cdim + 1, dim)
