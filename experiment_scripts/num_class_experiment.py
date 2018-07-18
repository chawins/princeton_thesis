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


path = './tmp/num_class_experiment/'
x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

# clean_acc = []
# adv_acc = []

# for a in range(10):
#     for b in range(a + 1, 10):

#         print('{}_{}'.format(a, b))
#         x_train_ab = x_train[[y == a or y == b for y in y_train]]
#         y_train_ab = y_train[[y == a or y == b for y in y_train]]
#         y_train_ab[y_train_ab == a] = 0
#         y_train_ab[y_train_ab == b] = 1
#         y_train_ab_cat = keras.utils.to_categorical(y_train_ab, 2)

#         x_test_ab = x_test[[y == a or y == b for y in y_test]]
#         y_test_ab = y_test[[y == a or y == b for y in y_test]]
#         y_test_ab[y_test_ab == a] = 0
#         y_test_ab[y_test_ab == b] = 1
#         y_test_ab_cat = keras.utils.to_categorical(y_test_ab, 2)

#         model = build_cnn_mnist_2cls()
#         model.fit(x_train_ab, y_train_ab_cat,
#                   batch_size=128,
#                   epochs=10,
#                   verbose=1,
#                   validation_data=(x_test_ab, y_test_ab_cat))
#         acc = model.evaluate(x_test_ab, y_test_ab_cat, verbose=0)
#         clean_acc.append(acc)
#         model.save_weights(path + 'weights_{}_{}.h5'.format(a, b))

#         acc = []
#         for n_step in range(5, 55, 5):
#             # Find adversarial examples with PGD
#             x_adv = PGD(model, x_test_ab, y_test_ab, grad_fn=None, norm="inf",
#                         n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
#             acc.append(model.evaluate(x_adv, y_test_ab_cat, verbose=0))
#         adv_acc.append(acc)

# pickle.dump(clean_acc, open(path + 'clean_acc.p', 'wb'))
# pickle.dump(adv_acc, open(path + 'adv_acc.p', 'wb'))

model = build_cnn_mnist()
model.load_weights('./tmp/weights/mnist_cnn_smxe.h5')
acc = []
for n_step in range(5, 55, 5):
    x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="inf",
                n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
    acc.append(model.evaluate(x_adv, y_test_cat, verbose=0))
pickle.dump(acc, open(path + '10cls_acc.p', 'wb'))
