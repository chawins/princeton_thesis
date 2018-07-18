# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import regularizers

x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)

model = build_cnn_mnist()


epochs = 2000
batch_size = 128
n_batch = len(x_train) // batch_size
min_loss = np.inf

start_time = 0

for epoch in range(epochs):
    start_time = time.time()
    ind = np.random.permutation(len(x_train))
    for batch in range(n_batch):
        b_ind = ind[batch * batch_size:(batch + 1) * batch_size]
        x, y, y_ = x_train[b_ind], y_train_cat[b_ind], y_train[b_ind]
        # PGD
        x_t = PGD(model, x, y_, grad_fn=None, norm="inf",
                  n_step=40, step_size=0.01, target=False,
                  init_rnd=0., proj=None, early_stop=False)

        model.train_on_batch(x_t, y)

    score = model.evaluate(x_test, y_test_cat, verbose=0)
    val_loss = score[0]
    if val_loss < min_loss:
        min_loss = val_loss
        model.save_weights('./tmp_adv.h5')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter == 5:
            break

    elasped_time = time.time() - start_time
    print('epoch: {} - {} (time: {:.2f}s)'.format(epoch, score, elasped_time))

model.load_weights('./tmp_adv.h5')
print(model.evaluate(x_train, y_train_cat, verbose=0))
print(model.evaluate(x_test, y_test_cat, verbose=0))
model.save_weights('./tmp/adv_train/weights_mnist.h5')
