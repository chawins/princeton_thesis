# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = keras.callbacks.ModelCheckpoint(
    './tmp.h5', save_best_only=True, save_weights_only=True, period=1)

path = './data/sphere/'
n = 1010000  # 70000
n_train = 1000000  # 60000
d = 500

r = 1
R = 1.3
x = np.random.normal(loc=0, scale=1, size=(n, d))
x /= np.linalg.norm(x, axis=1, keepdims=True)
x[:n//2] *= r
x[n//2:] *= R
y = np.concatenate(([0] * (n//2), [1] * (n//2)))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=n_train)
y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(1000, activation='linear', input_shape=(d,)))
model.add(Lambda(lambda x: x ** 2))
model.add(Dense(2, activation='linear'))
model.compile(loss=output_fn,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.fit(x_train, y_train_cat,
          batch_size=50,
          epochs=50,
          verbose=0,
          callbacks=[earlystop, checkpoint],
          validation_data=(x_test, y_test_cat))
model.load_weights('./tmp.h5')
print(model.evaluate(x_train, y_train_cat))
print(model.evaluate(x_test, y_test_cat))
model.save_weights('{}weights_d{}_1000-quad_1m.h5'.format(path, d))
