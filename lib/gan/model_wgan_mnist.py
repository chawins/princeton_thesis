"""
Implementation of Wasserstein GAN edited from 
https://github.com/bobchennan/Wasserstein-GAN-Keras/blob/master/mnist_wacgan.py
"""


import keras.backend as K
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, LeakyReLU, MaxPooling2D,
                          Multiply, Reshape, UpSampling2D)
from keras.models import Model, Sequential
from keras.constraints import max_norm
from param import *


CLIP_VALUE = 0.01
cnst = max_norm(CLIP_VALUE)


def modified_binary_crossentropy(target, output):
    return K.mean(target * output)


def build_generator():

    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    model = Sequential()

    model.add(Dense(1024, input_dim=LATENT_SIZE))
    model.add(LeakyReLU())
    model.add(Dense(128 * 7 * 7))
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 128)))

    # upsample to (..., 14, 14)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same',
                     kernel_initializer='glorot_uniform'))
    model.add(LeakyReLU())

    # upsample to (..., 28, 28)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same',
                     kernel_initializer='glorot_uniform'))
    model.add(LeakyReLU())

    # take a channel axis reduction
    model.add(Conv2D(1, (2, 2), padding='same', activation='tanh',
                     kernel_initializer='glorot_uniform'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(LATENT_SIZE, ))

    # this will be our label
    image_class = Input(shape=(1, ), dtype='int32')

    # 10 classes in MNIST
    embed = Embedding(
        10, LATENT_SIZE, embeddings_initializer='glorot_normal')(image_class)
    flatten = Flatten()(embed)

    # hadamard product between z-space and a class conditional embedding
    mul = Multiply()([latent, flatten])

    fake_image = model(mul)

    return Model(inputs=[latent, image_class], outputs=fake_image)


def build_discriminator():

    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    model = Sequential()
    #model.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
    model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                     input_shape=INPUT_SHAPE, kernel_constraint=cnst))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                     kernel_constraint=cnst))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2),
                     kernel_constraint=cnst))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1),
                     kernel_constraint=cnst))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())

    image = Input(shape=INPUT_SHAPE)

    features = model(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='linear', name='generation',
                 kernel_constraint=cnst)(features)
    aux = Dense(N_CLASSES, activation='softmax', name='auxiliary',
                kernel_constraint=cnst)(features)

    return Model(inputs=image, outputs=[fake, aux])


def combine_g_d(g, d):

    latent = Input(shape=(LATENT_SIZE, ))
    image_class = Input(shape=(1, ), dtype='int32')
    fake = g([latent, image_class])

    # we only want to be able to train generation for the combined model
    d.trainable = False
    dis, aux = d(fake)
    return Model(inputs=[latent, image_class], outputs=[dis, aux])
