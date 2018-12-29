import os
import random

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


class BasicCNNModel:
    log_dir = "logs/stage1"

    @classmethod
    def init(cls):
        cls.model = Sequential()
        cls.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(176, 200, 3), activation='relu'))
        cls.model.add(Conv2D(32, (3, 3), activation='relu'))
        cls.model.add(MaxPooling2D(pool_size=(2, 2)))
        cls.model.add(Dropout(0.2))
        cls.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        cls.model.add(Conv2D(64, (3, 3), activation='relu'))
        cls.model.add(MaxPooling2D(pool_size=(2, 2)))
        cls.model.add(Dropout(0.2))
        cls.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        cls.model.add(Conv2D(128, (3, 3), activation='relu'))
        cls.model.add(MaxPooling2D(pool_size=(2, 2)))
        cls.model.add(Dropout(0.2))
        cls.model.add(Flatten())
        cls.model.add(Dense(512, activation='relu'))
        cls.model.add(Dropout(0.5))
        cls.model.add(Dense(4, activation='softmax'))
        return cls
    
    @classmethod
    def compile(cls, lr):
        cls.model.compile(
            loss='categorical_crossentropy', 
            optimizer=keras.optimizers.adam(lr=lr, decay=1e-6),
            metrics=['accuracy']
        )
        return cls

    @classmethod
    def fit(cls, x_train, y_train, x_test, y_test, batch_size):
        cls.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1, 
            callbacks=[cls.get_tensorboard()]
        )
    
    @classmethod
    def get_tensorboard(cls):
        return TensorBoard(log_dir=cls.log_dir)

    @classmethod
    def get_model(cls):
        return cls.model

    @classmethod
    def save(cls, fn):
        cls.model.save(fn)

    @classmethod
    def load(cls, model_path):
        cls.model.load(model_path)
