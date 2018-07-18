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


x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

path = './tmp/mnist_hinge/'

adv_all = []
out_all = []


def experiment(mod):
    adv = []
    out = []
    x_adv = PGD(mod, x_test, y_test, grad_fn=None, norm="inf", n_step=50,
                step_size=0.01, target=False, init_rnd=0., early_stop=True, proj='img')
    ind = np.argmax(mod.predict(x_adv), axis=1) != y_test
    adv.append(np.linalg.norm((x_adv - x_test)[ind].reshape(-1, 784), axis=1))
    score = mod.evaluate(x_adv, y_test_cat)[1]
    out.append(score)

    x_adv = PGD(mod, x_test, y_test, grad_fn=None, norm="inf", n_step=100,
                step_size=0.01, target=False, init_rnd=0., early_stop=True, proj='img')
    ind = np.argmax(mod.predict(x_adv), axis=1) != y_test
    adv.append(np.linalg.norm((x_adv - x_test)[ind].reshape(-1, 784), axis=1))
    score = mod.evaluate(x_adv, y_test_cat)[1]
    out.append(score)

    x_adv = PGD(mod, x_test, y_test, grad_fn=None, norm="2", n_step=50,
                step_size=0.1, target=False, init_rnd=0., early_stop=True, proj='img')
    ind = np.argmax(mod.predict(x_adv), axis=1) != y_test
    adv.append(np.linalg.norm((x_adv - x_test)[ind].reshape(-1, 784), axis=1))
    score = mod.evaluate(x_adv, y_test_cat)[1]
    out.append(score)

    x_adv = PGD(mod, x_test, y_test, grad_fn=None, norm="2", n_step=100,
                step_size=0.1, target=False, init_rnd=0., early_stop=True, proj='img')
    ind = np.argmax(mod.predict(x_adv), axis=1) != y_test
    adv.append(np.linalg.norm((x_adv - x_test)[ind].reshape(-1, 784), axis=1))
    score = mod.evaluate(x_adv, y_test_cat)[1]
    out.append(score)

    adv_all.append(adv)
    out_all.append(out)
    pickle.dump(out_all, open(path + 'adv_acc.p', 'wb'))
    pickle.dump(adv_all, open(path + 'norm.p', 'wb'))


for m in ['a', 'b', 'c', 'd']:

    if m == 'a':
        model = build_cnn_mnist()
    elif m == 'b':
        model = build_cnn_mnist_2()
    elif m == 'c':
        model = build_dnn_mnist(784, 300, 4)
    else:
        model = build_dnn_mnist(784, 1200, 6)

    for C in ['1', '1e2', '1e4']:

        model.load_weights('./tmp/mnist_{}_hinge_c{}/model.h5'.format(m, C))
        experiment(model)
