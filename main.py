import os
import time
import argparse
import datetime

from .gamelauncher import GameLauncher
from .trainers import FullTrainer, AttackTrainer
from .mainbot import MainBot
from .loggers import logger

import numpy as np
from sc2 import Result

import tensorflow as tf
import keras.backend.tensorflow_backend as backend


MODEL_PATH = 'BasicCNN-10-epochs-0.0001-LR-STAGE1'


def get_session(gpu_fraction=0.85):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


backend.set_session(get_session())

parser = argparse.ArgumentParser(
    prog='main.py',
    description='Yuri, the StarCraft II bot'
)
parser.add_argument('--type')
parser.add_argument('--model', help='model path')
parser.add_argument('--difficulty', help='computer difficulty')
parser.add_argument('--race', help='computer race')
parser.add_argument(
    '--realtime', action='store_true',
    help='Run the game in realtime speed'
)

cmd_args = parser.parse_args()
game_type = cmd_args.type
model = cmd_args.model
realtime = cmd_args.realtime
difficulty = cmd_args.difficulty if cmd_args.difficulty is not None else 'medium'
race = cmd_args.race if cmd_args.race is not None else 'zerg'

model_type = 'attack'

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log.txt')

if str(game_type) == 'game':
    if model is None:
        game_launcher = GameLauncher(MainBot, False, None, realtime=realtime)
    else:
        game_launcher = GameLauncher(MainBot, True, model, realtime=realtime)

    train_data_tensor = list()
    game_result = game_launcher.start_game(difficulty, race, train_data_tensor)
    logger.debug(f'Game Result: {game_result}')

    if game_result == Result.Victory:
        np.save(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f'{model_type}_local_train/{str(int(time.time()))}.npy'),
            np.array(train_data_tensor))

    with open(log_path, 'a') as f:
        prefix = 'Model: ' if game_launcher.use_model else 'Random: '
        f.write(f'{datetime.datetime.now()}:{prefix}{game_result}\n')

elif str(game_type) == 'train':
    if model_type == 'attack':
        trainer = AttackTrainer()
        trainer.prepare_model(model)
        trainer.train()
    elif model_type == 'full':
        trainer = FullTrainer()
        trainer.prepare_model(model)
        trainer.train()
