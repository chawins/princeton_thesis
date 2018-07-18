"""
A run script to start adversarial training.
"""

import os
from os.path import basename

import keras
from keras import backend as K
from keras.models import save_model
from lib.attacks import symb_iter_fgs, symbolic_fgs
from lib.keras_utils import *
from lib.tf_utils import tf_test_error_rate, tf_train
from lib.utils import *
from parameters import *
from tensorflow.python.platform import flags

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Specify training set to load from ithin DATA_DIR
FLAGS = flags.FLAGS


def main():
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    flags.DEFINE_integer('NUM_EPOCHS', args.epochs, 'Number of epochs')

    # Get MNIST test data
    x_train, y_train, x_test, y_test = load_dataset_mnist()
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    x = K.placeholder(shape=(None, 28, 28, 1))
    y = K.placeholder(shape=(BATCH_SIZE, 10))

    eps = args.epsl
    x_advs = [None]

    #model = build_cnn_mnist()
    #model = build_cnn_mnist_2()
    #model = build_dnn_mnist(784, 300, 4)
    model = build_dnn_mnist(784, 1200, 6)

    if args.iter == 0:
        logits = model(x)
        grad = gen_grad(x, logits, y, loss='training')
        x_advs = symbolic_fgs(x, grad, eps=eps)
    elif args.iter == 1:
        x_advs = symb_iter_fgs(model, x, y, steps=40, alpha=0.01, eps=args.eps)

    # Train an MNIST model
    tf_train(x, y, model, x_train, y_train, x_advs=x_advs, benign=args.ben)

    # Finally print the result!
    test_error = tf_test_error_rate(model, x, x_test, y_test)
    print(test_error)

    # Specify model name
    model_name = './tmp/adv_train/weights_D_mnist_40-0.01.h5'
    # save_model(model, model_name)
    model.save_weights(model_name)
    # json_string = model.to_json()
    # with open(model_name + '.json', 'w') as f:
    #     f.write(json_string)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--norm", type=str, default='linf',
                        help="norm used to constrain perturbation")
    parser.add_argument("--iter", type=int, default=1,
                        help="whether an iterative training method is to be used")
    parser.add_argument("--ben", type=int, default=1,
                        help="whether benign data is to be used while performing adversarial training")

    args = parser.parse_args()
    main()
