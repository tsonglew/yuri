import os
import time
import argparse
import datetime

from .config import ConfigLoader
from .gamelauncher import GameLauncher
from .trainers import FullTrainer, AttackTrainer
from .mainbot import MainBot
from .loggers import logger

import numpy as np
from sc2 import Result

import tensorflow as tf
import keras.backend.tensorflow_backend as backend


def get_session(gpu_fraction=0.85):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


backend.set_session(get_session())

# load commondline args
parser = argparse.ArgumentParser(
    prog='main.py',
    description='Yuri, the StarCraft II bot'
)
parser.add_argument('--type')
cmd_args = parser.parse_args()
game_type = cmd_args.type

# load configuration file
cl = ConfigLoader('yuri.json')
configs = cl.get_json()
config_json = configs.get(game_type)
model_type = config_json.get('model_type')
model = config_json.get('model', None)


if str(game_type) == 'game':

    realtime = config_json.get('realtime', False)
    difficulty = config_json.get('difficulty', 'easy')
    race = config_json.get('race', 'zerg')

    map_name = config_json.get('map')
    logfile_name = config_json.get('logfn')

    model_path = os.path.join(os.path.dirname(__file__), model)
    game_launcher = GameLauncher(MainBot, model is not None, model_path, map_name=map_name, realtime=realtime)

    train_data_tensor = list()
    game_result = game_launcher.start_game(difficulty, race, train_data_tensor)
    logger.debug(f'Game Result: {game_result}')

    if game_result == Result.Victory:
        np.save(
            os.path.join(os.path.dirname(__file__), f'{model_type}_local_train/{str(int(time.time()))}.npy'),
            np.array(train_data_tensor)
        )

    log_path = os.path.join(os.path.dirname(__file__), logfile_name)
    with open(log_path, 'a') as f:
        prefix = 'Model: ' if game_launcher.use_model else 'Random: '
        f.write(f'{datetime.datetime.now()}:{prefix}{game_result}\n')

elif str(game_type) == 'train':

    if model_type == 'attack':
        trainer = AttackTrainer(config_json)
    elif model_type == 'full':
        trainer = FullTrainer(config_json)

    trainer.prepare_model(model)
    trainer.train()
