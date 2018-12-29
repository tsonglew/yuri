from model import BasicCNNModel
from utils import check_data, recursive_shuffle, load_attack_train_data

import os
import random
import argparse


parser = argparse.ArgumentParser(
    prog='train.py',
    usage='python3 %(prog) [--reuse]',
    description='Train model'
)
parser.add_argument(
    '--reuse', action='store_true',
    help='use a local model without rebuild a new one'
)
cmd_args = parser.parse()

# Training model config consts
test_size = 100
batch_size = 128
learning_rate = 0.0001
hm_epochs = 10
train_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'train_data'
)

if cmd_args.reuse:
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1'
    )
    BasicCNNModel.load(model_path)
else:
    BasicCNNModel.init()
BasicCNNModel.compile(lr=learning_rate)

for i in range(hm_epochs):
    current = 0
    increment = 200
    all_files = os.listdir(train_data_dir)
    total_files_num = len(all_files)
    print(f'Training file num: {total_files_num}')
    random.shuffle(all_files)
    
    while current <= total_files_num:
        print(f"Currently doing {current}:{current+increment}")

        # Read 200 files per hm_epoch
        x_train, y_train, x_test, y_test = load_attack_train_data(
            train_data_dir,
            all_files[current:current+increment],
            test_size
        )
        try:
            BasicCNNModel.fit(x_train, y_train, x_test, y_test, batch_size)
        except KeyboardInterrupt:
            ex = True
        BasicCNNModel.save(f'BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1')
        if ex:
            exit(0)
        current += increment
