import os
import random

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


class BasicCNNModel:
    log_dir = "logs/stage1"

    def init(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(176, 200, 3), activation='relu'))
        # self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        # self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, activation='softmax'))
        return self
    

    def compile(self, lr):
        self.model.compile(
            loss='categorical_crossentropy', 
            optimizer=keras.optimizers.adam(lr=lr, decay=1e-6),
            metrics=['accuracy']
        )
        return self


    def fit(self, x_train, y_train, x_test, y_test, batch_size):
        self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1, 
            callbacks=[self.get_tensorboard()]
        )
        return self
    

    def get_tensorboard(self):
        return TensorBoard(log_dir=self.log_dir)


    def get_model(self):
        return self.model


    def save(self, fn):
        self.model.save(fn)
        return self


    def load(self, model_path):
        self.model.load(model_path)
        return self
