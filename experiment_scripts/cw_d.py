# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *
from lib.OptCarlini import *

import numpy as np
import tensorflow as tf


x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

np.random.seed = 1234
ind = np.random.permutation(len(x_test))[:2000]
x_test = x_test[ind]
y_test = y_test[ind]
y_test_cat = y_test_cat[ind]

path = './tmp/mnist_hinge/'

adv_all = []
out_all = []


def experiment(mod, weight_path):

    opt = OptCarlini(mod, target=False, c=1e1, lr=0.1, init_scl=0.,
                     use_bound=False, loss_op=0, k=0, var_change=True,
                     use_mask=False)

    x_adv = np.zeros_like(x_test)
    norm = np.zeros(len(x_test))

    for i, (x, y) in enumerate(zip(x_test, y_test_cat)):
        x_adv[i], norm[i] = opt.optimize(x, y,
                                         weight_path,
                                         n_step=1000, prog=False)
    ind = np.argmax(mod.predict(x_adv), axis=1) != y_test
    adv_all.append(norm[ind])
    score = mod.evaluate(x_adv, y_test_cat)[1]
    out_all.append(score)
    pickle.dump(out_all, open(path + 'adv_acc_cw_d_2.p', 'wb'))
    pickle.dump(adv_all, open(path + 'norm_cw_d_2.p', 'wb'))


for m in ['d']:

    if m == 'a':
        model = build_cnn_mnist()
    elif m == 'b':
        model = build_cnn_mnist_2()
    elif m == 'c':
        model = build_dnn_mnist(784, 300, 4)
    else:
        model = build_dnn_mnist(784, 1200, 6)

    for C in ['1']:

        weight_path = './tmp/mnist_{}_hinge_c{}/model.h5'.format(m, C)
        model.load_weights(weight_path)
        experiment(model, weight_path)
