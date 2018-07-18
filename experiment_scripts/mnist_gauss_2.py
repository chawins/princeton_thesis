# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *

import numpy as np
import tensorflow as tf


x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

path = './tmp/mnist_gauss/'

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


model_A = build_cnn_mnist()
model_B = build_cnn_mnist_2()
model_C = build_dnn_mnist(784, 300, 4)
model_D = build_dnn_mnist(784, 1200, 6)

# model_A.load_weights('./tmp/weights/mnist_A.h5')
# model_B.load_weights('./tmp/weights/mnist_B.h5')
# model_C.load_weights('./tmp/weights/mnist_C.h5')
# model_D.load_weights('./tmp/weights/mnist_D.h5')

for i, model in enumerate([model_A, model_B, model_C, model_D]):

    for std in [0.1, 0.5, 1.0]:

        epochs = 100
        batch_size = 128
        n_batch = len(x_train) // batch_size
        min_loss = np.inf
        early_stop_counter = 0

        for epoch in range(epochs):
            ind = np.random.permutation(len(x_train))
            for batch in range(n_batch):
                b_ind = ind[batch * batch_size:(batch + 1) * batch_size]
                x, y = x_train[b_ind], y_train_cat[b_ind]
                # Add Gaussian noise
                x_t = x + np.random.normal(0, std, size=x.shape)

                model.train_on_batch(x_t, y)

            score = model.evaluate(x_test, y_test_cat, verbose=0)
            print('epoch: {} - {}'.format(epoch, score))
            val_loss = score[0]
            if val_loss < min_loss:
                min_loss = val_loss
                model.save_weights('./tmp_gauss2.h5')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter == 5:
                    break

        model.load_weights('./tmp_gauss2.h5')
        print(model.evaluate(x_train, y_train_cat, verbose=0))
        print(model.evaluate(x_test, y_test_cat, verbose=0))
        model.save_weights('{}weights_{}_s{}.h5'.format(path, i, std))

        experiment(model)


pickle.dump(out_all, open(path + 'adv_acc.p', 'wb'))
pickle.dump(adv_all, open(path + 'x_adv.p', 'wb'))
