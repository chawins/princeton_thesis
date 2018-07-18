
import argparse
import os
import pickle
from collections import defaultdict

import keras
import numpy as np
from sklearn.utils import shuffle
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils.generic_utils import Progbar
from lib.gan.model_acgan_mnist import *
from lib.gan.utils import *
from PIL import Image

# Set CUDA visible device to GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

save_path = './tmp/acgan_mnist/'

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0002
adam_beta_1 = 0.5

n_epoch = 50
batch_size = 128
latent_dim = 100


def train(prog=True):
    """
    Code is based on https://github.com/lukedeo/keras-acgan
    """

    # Load MNIST
    x_train, y_train, x_test, y_test = load_mnist()

    # Build model
    d = build_discriminator()
    g = build_generator(latent_dim)

    # Set up optimizers
    adam = Adam(lr=adam_lr, beta_1=adam_beta_1)

    # Set loss function and compile models
    g.compile(optimizer=adam, loss='binary_crossentropy')
    d.compile(optimizer=adam, loss=[
        'binary_crossentropy', 'sparse_categorical_crossentropy'])
    combined = combine_g_d(g, d, latent_dim)
    combined.compile(optimizer=adam, loss=[
        'binary_crossentropy', 'sparse_categorical_crossentropy'])

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    n_batch = int(x_train.shape[0] / batch_size)
    for epoch in range(n_epoch):
        print('Epoch {} of {}'.format(epoch + 1, n_epoch))
        progress_bar = Progbar(target=n_batch)

        epoch_g_loss = []
        epoch_d_loss = []

        # Shuffle training set
        x_train, y_train = shuffle(x_train, y_train)

        for index in range(n_batch):
            progress_bar.update(index)

            # ---------------- Train discriminator --------------------------- #
            # Generate samples from g
            z = np.random.normal(0, 1, (batch_size, latent_dim))
            # Sample some labels from p_c
            y_sampled = np.random.randint(0, 10, batch_size)
            x_g = g.predict([z, y_sampled.reshape((-1, 1))], verbose=0)

            # Combine with real samples
            x_real = x_train[index * batch_size:(index + 1) * batch_size]
            x_d = np.concatenate((x_real, x_g))
            y_d = np.array([1] * batch_size + [0] * batch_size)
            # Conditional (auxilary) labels
            y_real = y_train[index * batch_size:(index + 1) * batch_size]
            y_aux = np.concatenate((y_real, y_sampled), axis=0)

            epoch_d_loss.append(d.train_on_batch(x_d, [y_d, y_aux]))

            # ---------------- Train generator ------------------------------- #
            # Generate 2 * batch_size samples to match d's batch size
            z = np.random.normal(0, 1, (2 * batch_size, latent_dim))
            y_sampled = np.random.randint(0, 10, 2 * batch_size)
            y_g = np.ones(2 * batch_size)

            epoch_g_loss.append(combined.train_on_batch(
                [z, y_sampled.reshape((-1, 1))], [y_g, y_sampled]))

        print('\nTesting for epoch {}:'.format(epoch + 1))
        n_test = x_test.shape[0]

        # ---------------- Test discriminator -------------------------------- #
        z = np.random.normal(0, 1, (n_test, latent_dim))
        y_sampled = np.random.randint(0, 10, n_test)
        x_g = g.predict([z, y_sampled.reshape((-1, 1))], verbose=0)

        x_d = np.concatenate((x_test, x_g))
        y_d = np.array([1] * n_test + [0] * n_test)
        y_aux = np.concatenate((y_test, y_sampled), axis=0)

        d_test_loss = d.evaluate(x_d, [y_d, y_aux], verbose=0)
        d_train_loss = np.mean(np.array(epoch_d_loss), axis=0)

        # ---------------- Test generator ------------------------------------ #
        z = np.random.normal(0, 1, (2 * n_test, latent_dim))
        y_sampled = np.random.randint(0, 10, 2 * n_test)
        y_g = np.ones(2 * n_test)

        g_test_loss = combined.evaluate(
            [z, y_sampled.reshape((-1, 1))], [y_g, y_sampled], verbose=0)
        g_train_loss = np.mean(np.array(epoch_g_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(g_train_loss)
        train_history['discriminator'].append(d_train_loss)
        test_history['generator'].append(g_test_loss)
        test_history['discriminator'].append(d_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *d.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # Save weights every epoch
        g.save_weights("{}weight_g_epoch{:03d}.h5".format(save_path, epoch))
        d.save_weights("{}weight_d_epoch{:03d}.h5".format(save_path, epoch))

        # generate some digits to display
        noise = np.random.normal(0, 1, (100, latent_dim))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = g.predict(
            [noise, sampled_labels], verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            '{}plot_epoch{:03d}_generated.png'.format(save_path, epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prog", dest="prog", action="store_false")
    parser.set_defaults(prog=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(prog=args.prog)
