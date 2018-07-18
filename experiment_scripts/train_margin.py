# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from parameters import *
from lib.utils import *
from lib.attacks import *

import numpy as np
import tensorflow as tf

from keras.datasets import mnist

x_train, y_train, x_test, y_test = load_dataset_mnist()

y_train_cat = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test_cat = keras.utils.to_categorical(y_test, NUM_LABELS)


def get_weights(estimator):
    """
    Extract weights from TF Estimator. Only works with a simple CNN/DNN.
    """

    weights = []
    weight = []
    layer_names = estimator.get_variable_names()
    for layer_name in layer_names:
        if layer_name.endswith("kernel"):
            weight.insert(0, estimator.get_variable_value(layer_name))
            weights.append(weight)
            weight = []
        elif layer_name.endswith("bias"):
            weight.append(estimator.get_variable_value(layer_name))

    return weights


def load_weights(model, weights):
    """
    Set weights in Keras model with a list of weights.
    """

    i = 0
    for layer in model.layers:
        # Check if layer has trainable weights
        if not layer.trainable_weights:
            continue
        # Set weight
        layer.set_weights(weights[i])
        i += 1

    assert i == len(weights), "Number of layers mismatch."


for C in [1, 1e2, 1e4]:

    for lamda in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:

        def fn_mnist_A_margin(features, labels, mode):
            """Model function for CNN."""

            EPS = 1e-6

            # Input Layer
            input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[3, 3],
                activation=tf.nn.relu)

            # Convolutional Layer and pooling #2
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=[2, 2], strides=2)

            # Dropout #1
            drop1 = tf.layers.dropout(
                inputs=pool1, rate=0., training=(mode == tf.estimator.ModeKeys.TRAIN))

            # Dense Layer
            drop1_flat = tf.reshape(drop1, [-1, 12 * 12 * 64])
            dense = tf.layers.dense(
                inputs=drop1_flat, units=128, activation=tf.nn.relu)
            drop2 = tf.layers.dropout(
                inputs=dense, rate=0., training=(mode == tf.estimator.ModeKeys.TRAIN))

            # Logits Layer
            logits = tf.layers.dense(inputs=drop2, units=10, name="logits")

            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Structured hinge loss max{0, 1 - (y_label - y_max)}
            # Not so elegant way to index tensor with another tensor
            indices = tf.range(tf.shape(logits)[0])
            gather_ind = tf.stack([indices, labels], axis=1)
            y_label = tf.gather_nd(logits, gather_ind)
            # Get 2 largest outputs
            y_2max = tf.nn.top_k(logits, 2)[0]
            # Find y_max = max(z[i != y])
            i_max = tf.to_int32(tf.argmax(logits, axis=1))
            y_max = tf.where(tf.equal(labels, i_max), y_2max[:, 1],
                             y_2max[:, 0])
            loss = tf.reduce_sum(tf.maximum(0., C - y_label + y_max))

            # Add penalty term
            grad = tf.gradients(y_label, input_layer, name='grad')
            grad_reshape = tf.reshape(grad, shape=[-1, 28 * 28])
            # grad_norm = tf.reduce_sum(tf.square(grad_reshape), axis=1)
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_reshape), axis=1))
            # diff = tf.square(y_label - y_max)
            diff = tf.abs(y_label - y_max)
            # diff = y_label
            margin = tf.divide(grad_norm, diff + EPS)
            penalty = tf.reduce_sum(margin, name='penalty')
            loss += lamda * penalty

            # Calculate batch accuracy
            tmp = tf.cast(tf.equal(i_max, labels), dtype=tf.float32)
            accuracy = tf.reduce_mean(tmp, name="accuracy")

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        path = './tmp/mnist_0_margin_c{}_l{}/'.format(C, lamda)
        model_fn = fn_mnist_A_margin
        # path = './tmp/mnist_a_margin_c1e4/'
        # model_fn = fn_mnist_A_hinge
        model = build_cnn_mnist()

        mnist_classifier = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=path)
        tensors_to_log = {"accuracy": "accuracy", "penalty": "penalty"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        weight_path = path + 'model'
        n_epochs = 200
        stop_counter = 0

        min_test_loss = np.inf
        save_i = 0
        train_out = {'accuracy': [], 'loss': []}
        test_out = {'accuracy': [], 'loss': []}

        for i in range(n_epochs):

            if i % 2 is 0:
                try:
                    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={"x": x_train},
                        y=y_train,
                        batch_size=128,
                        num_epochs=2,
                        shuffle=True)

                    mnist_classifier.train(
                        input_fn=train_input_fn,
                        hooks=[logging_hook])

                    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={"x": x_train},
                        y=y_train,
                        num_epochs=1,
                        shuffle=False)
                    eval_results = mnist_classifier.evaluate(
                        input_fn=eval_input_fn)
                    train_out['accuracy'].append(eval_results['accuracy'])
                    train_out['loss'].append(eval_results['loss'])

                    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={"x": x_test},
                        y=y_test,
                        num_epochs=1,
                        shuffle=False)
                    eval_results = mnist_classifier.evaluate(
                        input_fn=eval_input_fn)
                    test_loss = eval_results['loss']
                    test_out['accuracy'].append(test_loss)
                    test_out['loss'].append(eval_results['loss'])

                    if test_loss < min_test_loss:
                        min_test_loss = test_loss
                        save_i = i
                        weights = get_weights(mnist_classifier)
                        load_weights(model, weights)
                        model.save_weights(weight_path + '.h5')
                        stop_counter = 0
                        print(
                            '=========== (-) Test loss: {:.5f} ============='.format(test_loss))
                    else:
                        stop_counter += 1
                        print(
                            '=========== (+) Test loss: {:.5f} ============='.format(test_loss))

                    if stop_counter == 5:
                        break

                except:
                    pass

        print('================ Finished in {} epochs ================'.format(i))
        pickle.dump(train_out, open(weight_path + '_train.p', 'wb'))
        pickle.dump(test_out, open(weight_path + '_test.p', 'wb'))

        model = build_cnn_mnist()
        model.load_weights(path + 'model.h5')
        train_test_acc = []
        train_test_acc.append(model.evaluate(x_train, y_train_cat)[1])
        train_test_acc.append(model.evaluate(x_test, y_test_cat)[1])

pickle.dump(train_test_acc, open('./tmp/mnist_margin/acc_0.p', 'wb'))
