# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.svm import LinearSVC


x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

path = './tmp/combine/'

adv_all = []
acc_all = []
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


model_A = build_cnn_mnist()
model_B = build_cnn_mnist_2()
model_C = build_dnn_mnist(784, 300, 4)
model_D = build_dnn_mnist(784, 1200, 6)

for i in [2, 3]:

    if i == 0:
        model = model_A
        model.load_weights('./tmp/weights/mnist_A.h5')
    elif i == 1:
        model = model_B
        model.load_weights('./tmp/weights/mnist_B.h5')
    elif i == 2:
        model = model_C
        model.load_weights('./tmp/weights/mnist_C.h5')
    else:
        model = model_D
        model.load_weights('./tmp/weights/mnist_D.h5')

    # input placeholder
    inp = model.input
    # Output of second-to-last layer
    out = model.layers[-2].output
    # evaluation functions
    eval_fnc = K.function([inp, K.learning_phase()], [out])

    n_feat = int(out.get_shape()[-1])
    x_nn_train = np.zeros((len(x_train), n_feat))
    x_nn_test = np.zeros((len(x_test), n_feat))

    for i, x in enumerate(x_train):
        x_in = x[np.newaxis, ...]
        layer_outs = eval_fnc([x_in, 0])
        x_nn_train[i] = layer_outs[0][0]

    for i, x in enumerate(x_test):
        x_in = x[np.newaxis, ...]
        layer_outs = eval_fnc([x_in, 0])
        x_nn_test[i] = layer_outs[0][0]

    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:

        clf = LinearSVC(C=C, loss='hinge')
        clf.fit(x_nn_train, y_train)
        weight = [clf.coef_.transpose(), clf.intercept_]
        model.layers[-1].set_weights(weight)
        acc = []
        acc.append(model.evaluate(x_train, y_train_cat)[1])
        acc.append(model.evaluate(x_test, y_test_cat)[1])
        acc_all.append(acc)
        model.save_weights('{}weights_{}_C{}.h5'.format(path, i, C))
        experiment(model)

pickle.dump(out_all, open(path + 'adv_acc.p', 'wb'))
pickle.dump(adv_all, open(path + 'norm.p', 'wb'))
pickle.dump(acc_all, open(path + 'acc.p', 'wb'))
