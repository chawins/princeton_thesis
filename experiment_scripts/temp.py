# Specify visible cuda device
import os
import pickle

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from lib.attacks import *
from lib.keras_utils import *
from lib.utils import *
from parameters import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

data_path = './data/gauss/'
out_path = './tmp/gauss/'
steps = np.array(range(10, 110, 10))
output = []

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = keras.callbacks.ModelCheckpoint(
    './tmp_gauss.h5', save_best_only=True, save_weights_only=True, period=1)


def cal_adv(mod):
    adv_acc = []
    for n_step in steps:
        x_adv = PGD(mod, x_test, y_test, grad_fn=None, norm="inf",
                    n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
        score = mod.evaluate(x_adv, y_test_cat)
        adv_acc.append(score[1])
    return adv_acc


d = 100
x_train, y_train, x_test, y_test = pickle.load(
    open('{}d{}.p'.format(data_path, d), 'rb'))
y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

# 0 hidden layers
model = Sequential()
model.add(Dense(2, input_dim=d, activation='linear'))
model.compile(loss=output_fn,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])
model.fit(x_train, y_train_cat,
          batch_size=50,
          epochs=100,
          verbose=1,
          callbacks=[earlystop, checkpoint],
          validation_data=(x_test, y_test_cat))
model.save_weights('{}weights_d{}_depth{}.h5'.format(out_path, d, 0))
# model.load_weights('{}weights_d{}_depth{}.h5'.format(out_path, d, 0))
output.append(cal_adv(model))

# 1, 2, 4, 8 hidden layers
for depth in [16]:
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    model.add(Dense(1000, input_dim=d, activation='relu'))
    model.add(BatchNormalization())
    for _ in range(depth - 1):
        model.add(Dense(1000, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dense(2, activation='linear'))
    model.compile(loss=output_fn,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    model.fit(x_train, y_train_cat,
              batch_size=50,
              epochs=100,
              verbose=1,
              callbacks=[earlystop, checkpoint],
              validation_data=(x_test, y_test_cat))
    model.save_weights('{}weights_d{}_depth{}.h5'.format(out_path, d, depth))
    # model.load_weights('{}weights_d{}_depth{}.h5'.format(out_path, d, depth))
    output.append(cal_adv(model))

# Width
# for width in [40, 200, 1000, 5000]:
#     model = Sequential()
#     model.add(Dense(width, input_dim=d, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(width, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(2, activation='linear'))
#     model.compile(loss=output_fn,
#                   optimizer=keras.optimizers.Adam(lr=1e-4),
#                   metrics=['accuracy'])
#     model.fit(x_train, y_train_cat,
#               batch_size=50,
#               epochs=50,
#               verbose=1,
#               callbacks=[earlystop],
#               validation_data=(x_test, y_test_cat))
#     model.save_weights('{}weights_d{}_width{}.h5'.format(out_path, d, width))
#     # model.load_weights('{}weights_d{}_width{}.h5'.format(out_path, d, width))
#     output.append(cal_adv(model))

pickle.dump([steps, output], open(
    '{}adv_acc_d{}_depth-width_2.p'.format(out_path, d), 'wb'))
