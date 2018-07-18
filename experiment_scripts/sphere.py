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
regs = [1e-7, 1e-8, 1e-9]
# regs_str = ['1e-4', '1e-5', '1e-6']
steps = [50, 100, 150]

# for reg, reg_str in zip(regs, regs_str):
for reg in regs:

    out = []
    adv = []
    model = Sequential()
    model.add(Dense(1000, activation='linear',
                    kernel_regularizer=regularizers.l2(reg),
                    input_shape=(d,)))
    model.add(Lambda(lambda x: x ** 2))
    model.add(Dense(2, activation='linear',
                    kernel_regularizer=regularizers.l2(reg)))
    model.compile(loss=output_fn,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    model.load_weights(
        '{}weights_d{}_1000-quad_reg{}.h5'.format(path, d, reg))

    # Off, 10 * 0.02
    x_adv = PGD(model, x_test, y_test, grad_fn=None, norm="2", n_step=100,
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

pickle.dump(out_all, open(path + 'adv_acc_reg_2.p', 'wb'))
pickle.dump(adv_all, open(path + 'adv_reg_2.p', 'wb'))
