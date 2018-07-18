import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from parameters import *
from keras import regularizers

#-------------------------------- GTSRB Model ---------------------------------#


def output_fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct,
                                                      logits=predicted)


def build_mltscl_gtsrb():
    """
    Build multiscale CNN. The last layer must be logits instead of softmax.
    Return a compiled Keras model.
    """

    # Regularization
    l2_reg = keras.regularizers.l2(L2_LAMBDA)

    # Build model
    inpt = keras.layers.Input(shape=IMG_SHAPE)
    conv1 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        128, (5, 5), padding='same', activation='relu')(pool2)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

    pool4 = keras.layers.MaxPooling2D(pool_size=(4, 4))(pool1)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(pool2)

    flat1 = keras.layers.Flatten()(pool4)
    flat2 = keras.layers.Flatten()(pool5)
    flat3 = keras.layers.Flatten()(pool3)

    merge = keras.layers.Concatenate(axis=-1)([flat1, flat2, flat3])
    dense1 = keras.layers.Dense(1024, activation='relu',
                                kernel_regularizer=l2_reg)(merge)
    drop4 = keras.layers.Dropout(rate=0.5)(dense1)
    output = keras.layers.Dense(
        OUTPUT_DIM, activation=None, kernel_regularizer=l2_reg)(drop4)
    model = keras.models.Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=output_fn, metrics=['accuracy'])

    return model


def build_cnn_gtsrb():
    """
    Build CNN. The last layer must be logits instead of softmax.
    Return a compiled Keras model.
    """

    l2_reg = keras.regularizers.l2(L2_LAMBDA)

    # Build model
    inpt = keras.layers.Input(shape=IMG_SHAPE)
    conv1 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    conv2 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(drop1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    conv4 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(drop3)
    drop4 = keras.layers.Dropout(rate=0.3)(conv4)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    flat = keras.layers.Flatten()(pool2)
    dense1 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(flat)
    drop5 = keras.layers.Dropout(rate=0.5)(dense1)
    dense2 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(drop5)
    drop6 = keras.layers.Dropout(rate=0.5)(dense2)
    output = keras.layers.Dense(
        OUTPUT_DIM, activation=None, kernel_regularizer=l2_reg)(drop6)
    model = keras.models.Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=output_fn, metrics=['accuracy'])

    return model

#-------------------------------- MNIST Model ---------------------------------#


def build_cnn_mnist(reg=None, lamda=0):

    if reg == 'l2':
        regularizer = regularizers.l2(lamda)
    elif reg == 'l1':
        regularizer = regularizers.l1(lamda)
    else:
        regularizer = None

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizer))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='linear', kernel_regularizer=regularizer))

    model.compile(loss=output_fn, optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    return model


def build_dnn_mnist(d, width, depth, reg=None, lamda=0):

    assert depth >= 1
    assert width >= 1

    if reg == 'l2':
        regularizer = regularizers.l2(lamda)
    elif reg == 'l1':
        regularizer = regularizers.l1(lamda)
    else:
        regularizer = None

    model = Sequential()
    model.add(Reshape((784,), input_shape=(28, 28, 1)))
    for _ in range(depth):
        model.add(Dense(width, activation='relu',
                        kernel_regularizer=regularizer))
        model.add(Dropout(0.5))
    model.add(Dense(10, activation='linear', kernel_regularizer=regularizer))
    model.compile(loss=output_fn,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    return model


def build_cnn_mnist_2(reg=None, lamda=0):
    """From Ensemble adversarial training: Attacks and defenses"""

    # model = Sequential()
    # model.add(Dropout(0.25, input_shape=(28, 28, 1)))
    # model.add(Conv2D(64, (8, 8), activation='relu'))
    # model.add(Conv2D(128, (6, 6), activation='relu'))
    # model.add(Conv2D(128, (5, 5), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(10, activation='linear'))

    if reg == 'l2':
        regularizer = regularizers.l2(lamda)
    elif reg == 'l1':
        regularizer = regularizers.l1(lamda)
    else:
        regularizer = None

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizer))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='linear', kernel_regularizer=regularizer))

    model.compile(loss=output_fn, optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    return model


def build_cnn_mnist_2cls():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear'))

    model.compile(loss=output_fn, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

#------------------------------- F-MNIST Model --------------------------------#


def build_vgg_fmnist(mean, std):
    """
    Code borrowed from
    http://danialk.github.io/blog/2017/09/29/range-of-convolutional
    -neural-networks-on-fashion-mnist-dataset/
    """
    def norm_input(x): return (x - mean) / std

    cnn = Sequential([
        Lambda(norm_input, input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu',
               padding='same', input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    cnn.compile(loss='sparse_categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=0.0001),
                metrics=['accuracy'])

    return cnn


#-------------------------- Model for synthetic data --------------------------#

def build_dnn_baseline(d):

    model = Sequential()
    model.add(Dense(1000, input_dim=d, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='linear'))
    model.compile(loss=output_fn,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    return model


def build_dnn_wd(d, width, depth, out_dim=2):

    assert depth >= 1
    assert width >= 1

    model = Sequential()
    model.add(Dense(width, input_dim=d, activation='relu'))
    model.add(BatchNormalization())
    for _ in range(depth - 1):
        model.add(Dense(width, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss=output_fn,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    return model


def build_dnn_dropout(d):

    model = Sequential()
    model.add(Dense(1000, input_dim=d, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(2, activation='linear'))
    model.compile(loss=output_fn,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    return model

#---------------------------------- Utility -----------------------------------#


def gradient_model(model):
    """Return gradient function of model's loss w.r.t. input"""

    outdim = model.output.get_shape()[1].value
    y_true = K.placeholder(shape=(outdim, ))
    loss = model.loss_functions[0](y_true, model.output)
    grad = K.gradients(loss, model.input)

    return K.function([model.input, y_true, K.learning_phase()], grad)


def gradient_fn(model, fn='softmax'):
    """Return gradient function of cross entropy loss w.r.t. input"""

    outdim = model.output.get_shape()[1].value
    y_true = K.placeholder(shape=(outdim, ))
    if fn == 'softmax':
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_true, logits=model.output)
    elif fn == 'mse':
        # loss = tf.losses.mean_squared_error(y_true, model.output)
        loss = tf.reduce_mean(tf.square(y_true - model.output),)
    grad = K.gradients(loss, model.input)

    return K.function([model.input, y_true, K.learning_phase()], grad)


def gradient_input(grad_fn, x, y):
    """Wrapper function to use gradient function more cleanly"""

    return grad_fn([x.reshape(INPUT_SHAPE), y, 0])[0][0]


def gen_adv_loss(logits, y, loss='logloss', mean=False):
    """
    Generate the loss function.
    """

    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = K.cast(K.equal(logits, K.max(logits, 1, keepdims=True)), "float32")
        y = y / K.sum(y, 1, keepdims=True)
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    elif loss == 'logloss':
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = K.mean(out)
    # else:
    #     out = K.sum(out)
    return out


def gen_grad(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    """

    adv_loss = gen_adv_loss(logits, y, loss)

    # Define gradient of loss wrt input
    grad = K.gradients(adv_loss, [x])[0]
    return grad
