from utils import check_data, recursive_shuffle

import os
import random

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


# Training model config consts
test_size = 100
batch_size = 128
learning_rate = 0.0001
hm_epochs = 10


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(176, 200, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer=keras.optimizers.adam(lr=learning_rate, decay=1e-6),
    metrics=['accuracy']
)
tensorboard = TensorBoard(log_dir="logs/stage1")
train_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'train_data'
)

for i in range(hm_epochs):
    current = 0
    increment = 200
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    print(f'Training file num: {maximum}')
    random.shuffle(all_files)
    
    while current <= maximum:
        print(f"Currently doing {current}:{current+increment}")
        no_attacks = []
        attack_closest_to_nexus = []
        attack_enemy_structures = []
        attack_enemy_start = []

        # Read 200 files per hm_epoch
        for file in all_files[current:current+increment]:
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
        random.shuffle(train_data)

        x_train = np.array([i[1] for i in train_data[:-test_size]])
        print(f'x_train original shape: {x_train.shape}')
        x_train = x_train.reshape(-1, 176, 200, 3)
        print(f'x_train shape after reshaping: {x_train.shape}')
        y_train = np.array([i[0] for i in train_data[:-test_size]])

        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
        y_test = np.array([i[0] for i in train_data[-test_size:]])

        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1, 
            callbacks=[tensorboard]
        )
        model.save(f'BasicCNN-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1')
        current += increment
