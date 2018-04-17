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

d = 500
data_path = './data/gauss/'
out_path = './tmp/gauss/'
steps = np.array(range(10, 110, 10))
wd = [(1000, 2), (1600, 4), (2000, 5)]
ratio = [0.01, 0.1, 0.5]
output = []
train = []


def cal_adv(mod):
    adv_acc = []
    for n_step in steps:
        x_adv = PGD(mod, x_test, y_test, grad_fn=None, norm="inf",
                    n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
        score = model.evaluate(x_adv, y_test_cat)
        adv_acc.append(score[1])
    return adv_acc


x_train, y_train, x_test, y_test = pickle.load(
    open('{}d{}.p'.format(data_path, d), 'rb'))
y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

for r in ratio:

    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]
    n_t = int(len(x_train) * r)
    x_t = np.concatenate(
        (x_train[y_train == 0][:n_t // 2], x_train[y_train == 1][:n_t // 2]))
    y_t = np.concatenate(
        (y_train[y_train == 0][:n_t // 2], y_train[y_train == 1][:n_t // 2]))
    y_t_cat = keras.utils.to_categorical(y_t)

    train_r = []
    output_r = []
    for (width, depth) in wd:
        model = build_dnn_wd(d, width, depth)
        model.fit(x_t, y_t_cat,
                  batch_size=50,
                  epochs=50,
                  verbose=1,
                  callbacks=[earlystop],
                  validation_data=(x_test, y_test_cat))
        score = model.evaluate(x_t, y_t_cat)
        score.extend(model.evaluate(x_test, y_test_cat))
        train_r.append(score)
        model.save_weights('{}weights_d{}_{}-{}_train{}.h5'.format(
            out_path, d, width, depth, r))
        # Attack
        output_r.append(cal_adv(model))
    output.append(output_r)
    train.append(train_r)

pickle.dump(train, open(
    '{}train-info_d{}_train.p'.format(out_path, d), 'wb'))
pickle.dump([steps, output], open(
    '{}adv_acc_d{}_train.p'.format(out_path, d), 'wb'))
