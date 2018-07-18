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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

data_path = './data/gauss/'
out_path = './tmp/gauss/'
steps = np.array(range(10, 110, 10))

for d in [100, 500]:

    x_train, y_train, x_test, y_test = pickle.load(
        open('{}d{}.p'.format(data_path, d), 'rb'))
    y_train_cat = keras.utils.to_categorical(y_train)
    y_test_cat = keras.utils.to_categorical(y_test)
    model = build_dnn_baseline(d)
    model.load_weights('{}weights_d{}_baseline.h5'.format(out_path, d))

    adv_acc = []
    for n_step in steps:
        x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="inf",
                    n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
        score = model.evaluate(x_adv, y_test_cat)
        adv_acc.append(score[1])

    pickle.dump([steps, adv_acc], open(
        '{}adv_acc_d{}_baseline.p'.format(out_path, d), 'wb'))
