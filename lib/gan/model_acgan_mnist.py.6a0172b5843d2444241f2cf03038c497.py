from keras.layers import (Conv2D, Dense, MaxPooling2D, Reshape, UpSampling2D,
                          Input, Embedding, Multiply, LeakyReLU, Dropout,
                          Activation, Flatten, Concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model


def build_generator():
    
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
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
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(100,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        input = multiply([noise, label_embedding])

        img = model(input)

        return Model([noise, label], img)
def build_discriminator():

    image = Input(shape=(28, 28, 1))

    model = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                   input_shape=(28, 28, 1))(image)
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

    return Model(inputs=image, outputs=[fake, aux])


def combine_g_d(g, d, latent_size):

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1, ), dtype='int32')
    fake = g([latent, image_class])

    # we only want to be able to train generation for the combined model
    d.trainable = False
    dis, aux = d(fake)
    return Model(inputs=[latent, image_class], outputs=[dis, aux])
