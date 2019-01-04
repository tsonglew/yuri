from .base_trainer import BaseTrainer
from ..models import FullCNNModel
from ..loggers import logger

import os
import random

import numpy as np


class FullTrainer(BaseTrainer):

    def __init__(self):
        BaseTrainer.__init__(self)

        self.name = 'FullTrainer'
        self.train_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'full_train_data'
        )
        self.model = FullCNNModel()

    def prepare_model(self, reuse):
        """
        create a basic cnn model or reload a model from path if reuse
        """
        if reuse:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE2'
            )
            self.load(model_path)
        else:
            self.model.init()
        self.model.compile(lr=self.learning_rate)
        return self

    def train(self):
        try:
            for i in range(self.hm_epochs):
                current = 0
                all_files = os.listdir(self.train_data_dir)
                maximum = len(all_files)
                random.shuffle(all_files)

                while current <= maximum:
                    origin_files = all_files[current: current + self.increment]
                    logger.debug(f'WORKING ON {current}:{current + self.increment}, EPOCH:{i}')
                    choices = self.prepare_choices(origin_files)
                    x_train, y_train, x_test, y_test = self.prepare_train_data(choices)
                    self.fit(x_train, y_train, x_test, y_test)
                    self.save("BasicCNN-5000-epochs-0.001-LR-STAGE2")
                    current += self.increment

        except KeyboardInterrupt:
            self.save("BasicCNN-5000-epochs-0.001-LR-STAGE2")

    def prepare_choices(self, origin_files) -> list:
        """
        just make the lengths of each kind of choices equal and shuffle data 
        of each kind of choice
        """
        loaded_choices = self.load_choices(origin_files)
        shuffled_choices = self.shuffle_choices(loaded_choices)
        return shuffled_choices

    def load_choices(self, files):
        """
        load choices from files and distinguish them with the choices made

        choices structure:
        [
            [ # all data of choice 0
                [ # one choice of choice 0
                    choices list like [1, 0, ..., 0, 0],
                    monitor image of shape (176, 200)
                ],
                ...
            ],
            [ # all data of choice 1
            ],
            ...
        ]
        """
        choices = [[] for i in range(0, 14)]

        for file in files:
            try:
                full_path = os.path.join(self.train_data_dir, file)
                data = np.load(full_path)
                data = list(data)
                for d in data:
                    choice = np.argmax(d[0])
                    choices[choice].append([d[0], d[1]])
            except Exception as e:
                logger.error(str(e))

        return choices

    def shuffle_choices(self, choices):
        """
        shuffle the images of a choice and truncate images list of all 
        choices to equivalent
        """
        lengths = self.check_data(choices)

        lowest_data = min(lengths)

        for choice, choice_d in enumerate(choices):
            random.shuffle(choices[choice])
            choices[choice] = choice_d[:lowest_data]

        self.check_data(choices)
        return choices

    def prepare_train_data(self, choices):
        """
        shuffle data of all kinds of choices
        """
        train_data = list()

        for choice in choices:
            for d in choice:
                train_data.append(d)

        random.shuffle(train_data)
        return self.pick_test_data(train_data)

    def pick_test_data(self, train_data):
        """
        pick train data and test data from prepared data set
        """
        x_train = np.array([i[1] for i in train_data[:-self.test_size]]).reshape(-1, 176, 200, 1)
        y_train = np.array([i[0] for i in train_data[:-self.test_size]])

        x_test = np.array([i[1] for i in train_data[-self.test_size:]]).reshape(-1, 176, 200, 1)
        y_test = np.array([i[0] for i in train_data[-self.test_size:]])
        return x_train, y_train, x_test, y_test

    @staticmethod
    def check_data(choices):
        total_data = 0

        lengths = list()
        for choice, choice_data in enumerate(choices):
            total_data += len(choices[choice])
            lengths.append(len(choices[choice]))

        logger.debug(f'Total data length now is: {total_data}')
        return lengths
