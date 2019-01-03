import time
import argparse
import datetime

from .gamelauncher import GameLauncher
from .trainer import Trainer
from .mainbot import MainBot
from .loggers import logger

import numpy as np
from sc2 import Result


MODEL_PATH = 'BasicCNN-10-epochs-0.0001-LR-STAGE1'


parser = argparse.ArgumentParser(
    prog='main.py',
    description='Yuri, the StarCraft II bot'
)
parser.add_argument('--type')
parser.add_argument(
    '--use_model', action='store_true',
    help='use a trained model'
)
parser.add_argument(
    '--reuse', action='store_true',
    help='use a local model without rebuild a new one'
)

cmd_args = parser.parse_args()

if str(cmd_args.type) == 'game':
    game_launcher = GameLauncher(MainBot, cmd_args.use_model, MODEL_PATH, realtime=False)
    game_result = game_launcher.start_game()

    logger.debug(f'Game Result: {game_result}')
    logger.debug(f'Using model: {cmd_args.use_model}')

    if game_result == Result.Victory:
        np.save(f'local_train/{str(int(time.time()))}.npy', 
                np.array(game_launcher.get_train_data()))

    with open('logs/log.txt', 'a') as f:
        prefix = 'Model: ' if game_launcher.use_model else 'Random: '
        f.write(f'{datetime.datetime.now()}:{prefix}{game_result}\n')

elif str(cmd_args.type) == 'train':
    trainer = Trainer()
    trainer.prepare_model(cmd_args.reuse)
    trainer.train()
