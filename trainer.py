from .model import BasicCNNModel
from .utils import check_data, recursive_shuffle, load_attack_train_data
from .loggers import logger

import os
import random


class Trainer:

    def __init__(self):
        self.test_size = 100
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.hm_epochs = 10
        self.increment = 200
        self.train_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'train_data'
        )
        self.model = BasicCNNModel()


    def prepare_model(self, reuse):
        if reuse:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1'
            )
            self.load(model_path)
        else:
            self.init()
        self.compile(lr=self.learning_rate)


    def train(self):
        for i in range(self.hm_epochs):
            current = 0

            all_files = os.listdir(self.train_data_dir)
            total_files_num = len(all_files)
            logger.info(f'Training file num: {total_files_num}')
            random.shuffle(all_files)
            
            while current <= total_files_num:
                logger.info(f"Currently doing {current}:{current+increment}")

                # Read 200 files per hm_epoch
                x_train, y_train, x_test, y_test = load_attack_train_data(
                    self.train_data_dir,
                    all_files[current:current+self.increment],
                    self.test_size
                )
                try:
                    self.model.fit(x_train, y_train, x_test, y_test, self.batch_size)
                except KeyboardInterrupt:
                    ex = True
                self.model.save(f'BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1')
                if ex:
                    exit(0)
                current += self.increment
