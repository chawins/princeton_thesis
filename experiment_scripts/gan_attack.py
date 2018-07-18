# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from parameters import *
from lib.utils import *
from lib.attacks import *

import numpy as np
import tensorflow as tf

x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test_cat = keras.utils.to_categorical(y_test, NUM_LABELS)


def random_sample(size):

    # Generate samples from g
    z = np.random.normal(0, 1, size)
    # Sampled labels
    y_sampled = np.random.randint(0, 10, size[0])
    return z, y_sampled


def generate_random(g, size):

    z, y_sampled = random_sample(size)
    x_g = g.predict([z, y_sampled.reshape((-1, 1))], verbose=0)
    return x_g, y_sampled


def collage(images):
    img = (np.concatenate([np.concatenate([s for s in r], axis=1)
                           for r in np.split(images, 10)], axis=0) *
           127.5 + 127.5).astype(np.uint8)
    return np.squeeze(img)


def show(x):
    plt.imshow(x.reshape(28, 28) / 2 + 1, cmap='gray')
    plt.axis('off')
    plt.show()


def grad_acgan_cross_entropy(model):
    y_true = K.placeholder(shape=(OUTPUT_DIM, ))
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=model.output)
    grad = K.gradients(loss, model.input[0])
    return K.function([model.input[0], model.input[1],
                       y_true, K.learning_phase()], grad)


def grad_acgan_hinge(model):

    labels = K.placeholder(shape=(OUTPUT_DIM, ), dtype=tf.int32)
    logits = model.output[0]
    i_label = tf.to_int32(tf.argmax(labels))
    y_label = logits[i_label]
    # Get 2 largest outputs
    y_2max = tf.nn.top_k(logits, 2)[0]
    # Find y_max = max(z[i != y])
    i_max = tf.to_int32(tf.argmax(logits))
    y_max = tf.where(tf.equal(i_label, i_max),
                     y_2max[1], y_2max[0])
    loss = tf.maximum(0., 1. - y_label + y_max)
    grad = K.gradients(loss, model.input[0])
    return K.function([model.input[0], model.input[1],
                       labels, K.learning_phase()], grad)


def PGD(model, x, y, grad_fn=None, norm="2", n_step=40, step_size=0.05,
        target=True, init_rnd=0.1):
    """
    PGD attack with random start
    """

    EPS = 1e-6
    x_adv = np.zeros_like(x)
    y_cat = keras.utils.to_categorical(y, NUM_LABELS)

    for i, x_cur in enumerate(x):
        epsilon = np.random.uniform(size=x_cur.shape) - 0.5
        if norm == "2":
            try:
                epsilon /= (np.linalg.norm(epsilon) + EPS)
            except ZeroDivisionError:
                raise
        elif norm == "inf":
            epsilon = np.sign(epsilon)
        else:
            raise ValueError("Invalid norm!")

        x_adv[i] = x_cur + init_rnd * epsilon

    if not grad_fn:
        grad_fn = gradient_fn(model)
    start_time = time.time()

    for i, x_in in enumerate(x_adv):
        print(i)
        x_cur = np.copy(x_in)
        # Start update in steps
        for _ in range(n_step):
            grad = grad_fn([x_in.reshape(1, -1), y[i].reshape(1, -1),
                            y_cat[i], 0])[0][0]
            if target:
                grad *= -1
            if norm == "2":
                try:
                    grad /= (np.linalg.norm(grad) + EPS)
                except ZeroDivisionError:
                    raise
            elif norm == "inf":
                grad = np.sign(grad)
            else:
                raise ValueError("Invalid norm!")

            x_cur += grad * step_size
            loss = model.evaluate([x_cur[np.newaxis], y[i, np.newaxis]],
                                  y[i, np.newaxis], verbose=0)[0]
#             out = model.predict([x_cur[np.newaxis], y[i, np.newaxis]],
#                                 verbose=0)[0]
#             if np.argmax(out) != y[i]:
#                 loss = np.maximum(0, 1 - out[y[i]] + np.sort(out)[-1])
#             else:
#                 loss = np.maximum(0, 1 - out[y[i]] + np.sort(out)[-2])
            print(loss)

        x_adv[i] = np.copy(x_cur)

        # Progress printing
        if (i % 200 == 0) and (i > 0):
            elasped_time = time.time() - start_time
            print("Finished {} samples in {:.2f}s.".format(i, elasped_time))
            start_time = time.time()

    return x_adv


model = build_cnn_mnist()
# model.load_weights('./tmp/weights/mnist_cnn_hinge.h5')
model.load_weights('./tmp/weights/mnist_cnn_smxe.h5')
# model.load_weights('./tmp/mnist_cnn_margin_C1_L1/model.h5')
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

from lib.gan.model_acgan_mnist import *

latent_dim = 100
d = build_discriminator()
g = build_generator(latent_dim)
d.load_weights('./tmp/acgan_mnist/weight_d_epoch049.h5')
g.load_weights('./tmp/acgan_mnist/weight_g_epoch049.h5')

latent = Input(shape=(latent_dim, ))
image_class = Input(shape=(1, ), dtype='int32')
img = g([latent, image_class])
y = model(img)
combine = Model(inputs=[latent, image_class], outputs=y)
combine.trainable = False
combine.compile(loss=keras.losses.sparse_categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])


z, y = random_sample((1000, latent_dim))
y_cat = keras.utils.to_categorical(y, NUM_LABELS)
grad_fn = grad_acgan_cross_entropy(combine)
# grad_fn = grad_acgan_hinge(combine)


x_adv = PGD(combine, z, y, grad_fn=grad_fn, norm="2", n_step=200,
            step_size=0.01, target=False, init_rnd=0.)
pickle.dump(x_adv, open('./tmp/gan/x_adv_pgd.p', 'wb'))


x = g.predict([z, y])
y_pred = np.argmax(model.predict(x), axis=1)
y_adv = np.argmax(combine.predict([x_adv, y]), axis=1)

n_ben = np.sum(y_pred == y)
n_adv = np.sum(np.logical_and(y_pred == y, y_adv != y))
suc_rate = n_adv / n_ben
print(n_ben, n_adv, suc_rate)
