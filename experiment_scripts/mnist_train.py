# Specify visible cuda device
import os
import pickle
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from lib.attacks import *
from lib.keras_utils import *
from lib.utils import *
from parameters import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

d = 784
width = 1600
depth = 4
data_path = './data/gauss/'
out_path = './tmp/gauss/'
steps = np.array(range(10, 110, 10))
output = []
train = []


n = 70000
n_train = 60000

# mean_a = np.zeros(d)
# mean_b = np.ones(d)
# std = np.sqrt(d) / 6.  # touching at 3 std
# x_a = np.random.normal(loc=mean_a, scale=std, size=(int(n/2), d))
# x_b = np.random.normal(loc=mean_b, scale=std, size=(int(n/2), d))
# x = np.concatenate([x_a, x_b])
# y = np.concatenate(([0] * int(n/2), [1] * int(n/2)))
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n_train)
# y_train_cat = keras.utils.to_categorical(y_train)
# y_test_cat = keras.utils.to_categorical(y_test)

# # -----------------------------------------------------------------------------

# r = 1
# model = build_dnn_wd(d, width, depth)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# model.fit(x_train, y_train_cat,
#           batch_size=50,
#           epochs=50,
#           verbose=1,
#           callbacks=[earlystop],
#           validation_data=(x_test, y_test_cat))
# score = model.evaluate(x_train, y_train_cat)
# score.extend(model.evaluate(x_test, y_test_cat))
# train.append(score)
# model.save_weights('{}weights_d{}_{}-{}_train{}.h5'.format(
#     out_path, d, width, depth, r))

# adv = []
# for n_step in steps:
#     x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="inf",
#                 n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
#     score = model.evaluate(x_adv, y_test_cat)
#     adv.append(score[1])
# output.append(adv)

# pickle.dump(train, open(
#     '{}train-info_mnist_train.p'.format(out_path), 'wb'))
# pickle.dump([steps, output], open(
#     '{}adv_acc_mnist_train.p'.format(out_path), 'wb'))

# # -----------------------------------------------------------------------------

# r = 0.1
# p = np.random.permutation(len(x_train))
# x_train, y_train = x_train[p], y_train[p]
# n_t = int(len(x_train) * r)
# x_t = np.concatenate(
#     (x_train[y_train == 0][:n_t // 2], x_train[y_train == 1][:n_t // 2]))
# y_t = np.concatenate(
#     (y_train[y_train == 0][:n_t // 2], y_train[y_train == 1][:n_t // 2]))
# y_t_cat = keras.utils.to_categorical(y_t)

# model = build_dnn_wd(d, width, depth)
# model.fit(x_t, y_t_cat,
#           batch_size=50,
#           epochs=50,
#           verbose=1,
#           callbacks=[earlystop],
#           validation_data=(x_test, y_test_cat))
# score = model.evaluate(x_t, y_t_cat)
# score.extend(model.evaluate(x_test, y_test_cat))
# train.append(score)
# model.save_weights('{}weights_d{}_{}-{}_train{}.h5'.format(
#     out_path, d, width, depth, r))

# adv = []
# for n_step in steps:
#     x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="inf",
#                 n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
#     score = model.evaluate(x_adv, y_test_cat)
#     adv.append(score[1])
# output.append(adv)

# pickle.dump(train, open(
#     '{}train-info_mnist_train.p'.format(out_path), 'wb'))
# pickle.dump([steps, output], open(
#     '{}adv_acc_mnist_train.p'.format(out_path), 'wb'))

# -----------------------------------------------------------------------------

x_train, y_train, x_test, y_test = load_dataset_mnist()
y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

# -----------------------------------------------------------------------------

# r = 1
# model = build_dnn_wd(d, width, depth, out_dim=10)
# model.fit(x_train, y_train_cat,
#           batch_size=50,
#           epochs=50,
#           verbose=1,
#           callbacks=[earlystop],
#           validation_data=(x_test, y_test_cat))
# score = model.evaluate(x_train, y_train_cat)
# score.extend(model.evaluate(x_test, y_test_cat))
# train.append(score)
# model.save_weights('{}weights_mnist_{}-{}_train{}.h5'.format(
#     out_path, width, depth, r))

# adv = []
# for n_step in steps:
#     x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="inf",
#                 n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
#     score = model.evaluate(x_adv, y_test_cat)
#     adv.append(score[1])
# output.append(adv)

# pickle.dump(train, open(
#     '{}train-info_mnist_train.p'.format(out_path), 'wb'))
# pickle.dump([steps, output], open(
#     '{}adv_acc_mnist_train.p'.format(out_path), 'wb'))

# -----------------------------------------------------------------------------

r = 0.1
p = np.random.permutation(len(x_train))
x_train, y_train = x_train[p], y_train[p]
n_t = int(len(x_train) * r)
ind = np.array([])
for i in range(10):
    ind = np.concatenate((ind, np.where(y_train == i)[0][:n_t // 10]))
ind = ind.astype(np.int32)
x_t, y_t = x_train[ind], y_train[ind]
y_t_cat = keras.utils.to_categorical(y_t)

model = build_dnn_wd(d, width, depth, out_dim=10)
model.fit(x_t, y_t_cat,
          batch_size=50,
          epochs=50,
          verbose=1,
          callbacks=[earlystop],
          validation_data=(x_test, y_test_cat))
score = model.evaluate(x_t, y_t_cat)
score.extend(model.evaluate(x_test, y_test_cat))
train.append(score)
model.save_weights('{}weights_mnist_{}-{}_train{}.h5'.format(
    out_path, width, depth, r))

adv = []
for n_step in steps:
    x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="inf",
                n_step=n_step, step_size=0.01, target=False, init_rnd=0.)
    score = model.evaluate(x_adv, y_test_cat)
    adv.append(score[1])
output.append(adv)

pickle.dump(train, open(
    '{}train-info_mnist_train.p'.format(out_path), 'wb'))
pickle.dump([steps, output], open(
    '{}adv_acc_mnist_train.p'.format(out_path), 'wb'))
