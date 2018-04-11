from keras.layers import (Conv2D, Dense, MaxPooling2D, Reshape, UpSampling2D,
                          Input, Embedding, multiply, LeakyReLU, Dropout,
                          Activation, Flatten, Concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model


def build_generator(latent_dim):

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(10, 100)(label))

    input = multiply([noise, label_embedding])

    img = model(input)

    return Model([noise, label], img)


def build_discriminator():

    img = Input(shape=(28, 28, 1))

    model = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                   input_shape=(28, 28, 1))(img)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(model)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(model)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(model)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Flatten()(model)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(model)
    aux = Dense(10, activation='softmax', name='auxiliary')(model)

    return Model(img, [fake, aux])


def combine_g_d(g, d, latent_dim):

    latent = Input(shape=(latent_dim, ))
    image_class = Input(shape=(1, ), dtype='int32')
    img = g([latent, image_class])

    # we only want to be able to train generation for the combined model
    d.trainable = False
    fake, aux = d(img)
    return Model(inputs=[latent, image_class], outputs=[fake, aux])
