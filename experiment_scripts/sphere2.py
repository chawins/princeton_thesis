# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *

import numpy as np
import tensorflow as tf
from keras import regularizers

data_path = './data/sphere/'
path = './tmp/sphere/'
d = 500
x_train, y_train, x_test, y_test = pickle.load(
    open('{}sphere_d{}.p'.format(data_path, d), 'rb'))
y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

out_all = []
adv_all = []
steps = [50, 100, 150]

out = []
adv = []
# model = Sequential()
# model.add(Dense(1000, activation='linear', input_shape=(d,)))
# model.add(Lambda(lambda x: x ** 2))
# model.add(Dense(2, activation='linear'))
# model.compile(loss=output_fn,
#               optimizer=keras.optimizers.Adam(lr=1e-4),
#               metrics=['accuracy'])
# model.load_weights('{}weights_d{}_1000-quad.h5'.format(path, d))
model = build_dnn_baseline(500)
model.load_weights('./tmp/sphere/weights_d500_1000-2.h5')

# Off, 10 * 0.02
x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="2", n_step=10,
            step_size=0.02, target=False, init_rnd=0., early_stop=True)
adv.append(x_adv)
score = model.evaluate(x_adv, y_test_cat)[1]
out.append(score)

# In, step * 0.001
for step in steps:
    x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="2", n_step=step,
                step_size=0.001, target=False, init_rnd=0., early_stop=True, proj='l2')
    adv.append(x_adv)
    score = model.evaluate(x_adv, y_test_cat)[1]
    out.append(score)

# model = Sequential()
# model.add(Dense(1000, activation='linear', input_shape=(d,)))
# model.add(Lambda(lambda x: x ** 2))
# model.add(Dense(2, activation='linear'))
# model.compile(loss=output_fn,
#               optimizer=keras.optimizers.Adam(lr=1e-4),
#               metrics=['accuracy'])
# model.load_weights('{}weights_d{}_1000-quad_1m.h5'.format(path, d))
model = Sequential()
model.add(Lambda(lambda x: x ** 2, input_shape=(d,)))
model.add(Dense(2, activation='linear'))
model.compile(loss=output_fn,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])
model.load_weights('./tmp/sphere/weights_d500_quad.h5')

# Off, 10 * 0.02
x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="2", n_step=10,
            step_size=0.02, target=False, init_rnd=0., early_stop=True)
adv.append(x_adv)
score = model.evaluate(x_adv, y_test_cat)[1]
out.append(score)

# In, step * 0.001
for step in steps:
    x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="2", n_step=step,
                step_size=0.001, target=False, init_rnd=0., early_stop=True, proj='l2')
    adv.append(x_adv)
    score = model.evaluate(x_adv, y_test_cat)[1]
    out.append(score)

adv_all.append(adv)
out_all.append(out)

pickle.dump(out_all, open(path + 'adv_acc_base.p', 'wb'))
pickle.dump(adv_all, open(path + 'adv_base.p', 'wb'))
