import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial

from home.pn.PycharmProjects.autoencoders.helper_functions import *


def plot_reconstructions(originals, reconstructions):
    plt.figure(figsize=(10, 4), dpi=100)
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(originals[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()
        # display reconstruction
        ax = plt.subplot(2, 10, i + 10 + 1)
        plt.imshow(reconstructions[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()
    plt.show()

if __name__ == '__main__':

    experimental_tasks1 = []
    settings = [(5, 0.0001), (10, 0.0001), (15, 0.0001)]
    train_data, test_data, m = get_mnist_data()

    for (num_epochs, l2_reg) in settings:

        n = 28 * 28  # for MNIST

        learning_rate = 0.01
        l2_reg = 0.0001

        X = tf.placeholder(tf.float32, shape=[None, n])

        he_init = tf.contrib.layers.variance_scaling_initializer()
        l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_regularizer)

        hidden1 = my_dense_layer(X, 300)
        hidden2 = my_dense_layer(hidden1, 150)
        hidden3 = my_dense_layer(hidden2, 300)
        outputs = my_dense_layer(hidden3, n, activation=None) ##Overwrite: no activation fn in last layer

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        #Train
        train_loss, test_loss = [], []
        #num_epochs = 5
        batch_size = 150
        with tf.Session()   as sess:
            init.run()
            train_loss.append(loss.eval(session=sess, feed_dict={X: train_data}))
            test_loss.append(loss.eval(session=sess, feed_dict={X: test_data}))
            for epoch in range(num_epochs):
                print("Epoch:", epoch, "/", num_epochs)
                n_batches = m//batch_size
                for batch in range(n_batches):
                    X_batch = train_data[batch*batch_size: (batch + 1)*batch_size]
                    sess.run(training_op, feed_dict={X: X_batch})
                train_loss.append(loss.eval(session=sess, feed_dict={X: train_data}))
                test_loss.append(loss.eval(session=sess, feed_dict={X: test_data}))

            reconstructions = outputs.eval(feed_dict={X: test_data[0: 10]})

        experimental_tasks1.append(((num_epochs, l2_reg), train_loss, test_loss))

        #plot_reconstructions(test_data[0: 10], reconstructions)

    plot_learning_curves([experimental_tasks1])