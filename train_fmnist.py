"""
Train Fashion MNIST CNN
Code borrowed from http://danialk.github.io/blog/2017/09/29/range-of-
convolutional-neural-networks-on-fashion-mnist-dataset/
"""
# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from lib.keras_utils import build_vgg_fmnist
from lib.utils import load_dataset_fmnist

batch_size = 512

# Load f-mnist, find mean and std
x_train, y_train, x_test, y_test = load_dataset_fmnist()
mean = x_train.mean().astype(np.float32)
std = x_train.std().astype(np.float32)

# Build Keras model
cnn = build_vgg_fmnist(mean, std)

# Data augmentation
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08,
                         shear_range=0.3, height_shift_range=0.08,
                         zoom_range=0.08)
batches = gen.flow(x_train, y_train, batch_size=batch_size)
val_batches = gen.flow(x_test, y_test, batch_size=batch_size)

cnn.fit_generator(batches,
                  steps_per_epoch=60000//batch_size,
                  epochs=50,
                  validation_data=val_batches,
                  validation_steps=10000//batch_size,
                  use_multiprocessing=True)

score = cnn.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save trained weight
cnn.save_weights('./tmp/weights/fmnist_vgg_smxe.h5')
