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


x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

path = './tmp/mnist_reg/'

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


for reg in ['l1']:

    if reg == 'l2':
        L = [1e-2, 1e-4, 1e-6]
    else:
        L = [1e-3, 1e-5, 1e-7]

    for lamda in L:

        model = build_cnn_mnist(reg=reg, lamda=lamda)

        # earlystop = keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=5)
        # checkpoint = keras.callbacks.ModelCheckpoint(
        #     './tmp_reg.h5', save_best_only=True, save_weights_only=True, period=1)
        # model.fit(x_train, y_train_cat,
        #           batch_size=128,
        #           epochs=100,
        #           verbose=1,
        #           callbacks=[earlystop, checkpoint],
        #           validation_data=(x_test, y_test_cat))
        # model.load_weights('./tmp_reg.h5')
        # print(model.evaluate(x_train, y_train_cat))
        # print(model.evaluate(x_test, y_test_cat))
        # model.save_weights('{}weights_0_{}_L{}.h5'.format(path, reg, lamda))
        model.load_weights('{}weights_0_{}_L{}.h5'.format(path, reg, lamda))
        experiment(model)

        model = build_cnn_mnist_2(reg=reg, lamda=lamda)

        # earlystop = keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=5)
        # checkpoint = keras.callbacks.ModelCheckpoint(
        #     './tmp_reg.h5', save_best_only=True, save_weights_only=True, period=1)
        # model.fit(x_train, y_train_cat,
        #           batch_size=128,
        #           epochs=100,
        #           verbose=1,
        #           callbacks=[earlystop, checkpoint],
        #           validation_data=(x_test, y_test_cat))
        # model.load_weights('./tmp_reg.h5')
        # print(model.evaluate(x_train, y_train_cat))
        # print(model.evaluate(x_test, y_test_cat))
        # model.save_weights('{}weights_1_{}_L{}.h5'.format(path, reg, lamda))
        model.load_weights('{}weights_1_{}_L{}.h5'.format(path, reg, lamda))
        experiment(model)

        model = build_dnn_mnist(784, 300, 4, reg=reg, lamda=lamda)

        # earlystop = keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=5)
        # checkpoint = keras.callbacks.ModelCheckpoint(
        #     './tmp_reg.h5', save_best_only=True, save_weights_only=True, period=1)

        # model.fit(x_train, y_train_cat,
        #           batch_size=128,
        #           epochs=100,
        #           verbose=1,
        #           callbacks=[earlystop, checkpoint],
        #           validation_data=(x_test, y_test_cat))
        # model.load_weights('./tmp_reg.h5')
        # print(model.evaluate(x_train, y_train_cat))
        # print(model.evaluate(x_test, y_test_cat))
        # model.save_weights('{}weights_2_{}_L{}.h5'.format(path, reg, lamda))
        model.load_weights('{}weights_2_{}_L{}.h5'.format(path, reg, lamda))
        experiment(model)

        model = build_dnn_mnist(784, 1200, 6, reg=reg, lamda=lamda)

        # earlystop = keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=5)
        # checkpoint = keras.callbacks.ModelCheckpoint(
        #     './tmp_reg.h5', save_best_only=True, save_weights_only=True, period=1)
        # model.fit(x_train, y_train_cat,
        #           batch_size=128,
        #           epochs=100,
        #           verbose=1,
        #           callbacks=[earlystop, checkpoint],
        #           validation_data=(x_test, y_test_cat))
        # model.load_weights('./tmp_reg.h5')
        # print(model.evaluate(x_train, y_train_cat))
        # print(model.evaluate(x_test, y_test_cat))
        # model.save_weights('{}weights_3_{}_L{}.h5'.format(path, reg, lamda))
        model.load_weights('{}weights_3_{}_L{}.h5'.format(path, reg, lamda))
        experiment(model)

pickle.dump(out_all, open(path + 'adv_acc_5.p', 'wb'))
pickle.dump(adv_all, open(path + 'norm_5.p', 'wb'))
