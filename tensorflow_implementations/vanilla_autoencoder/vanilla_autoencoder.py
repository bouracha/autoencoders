from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from functools import partial

#import mnist
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

data, labels = mnist["data"], mnist["target"]
m = data.shape[0]

print(m)
print(data.shape)
print(labels.shape)

some_digit = data[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")

#plt.axis("off")
#plt.show()

if __name__ == '__main__':

    n_inputs = 28 * 28  # for MNIST
    n_hidden1 = 300
    n_hidden2 = 150
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = 0.01
    l2_reg = 0.0001

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])

    he_init = tf.contrib.layers.variance_scaling_initializer()
    l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    ## Partial allows to use the function my_dense_layer with same set parameters each time
    my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_regularizer)

    hidden1 = my_dense_layer(X, n_hidden1)
    hidden2 = my_dense_layer(hidden1, n_hidden2)
    hidden3 = my_dense_layer(hidden2, n_hidden3)
    outputs = my_dense_layer(hidden3, n_outputs, activation=None) ##Overwrite: no activation fn in last layer

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss] + reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    n_epochs = 5
    batch_size = 150
    with tf.Session()   as sess:
        init.run()
        for epoch in range(n_epochs):
            print("Epoch:", epoch, "/", n_epochs)
            n_batches = m//batch_size
            #n_batches = mnist.train.num_examples
            for batch in range(n_batches):
                X_batch, y_batch = data[batch*batch_size: (batch + 1)*batch_size], labels[batch*batch_size: (batch + 1)*batch_size]
                sess.run(training_op, feed_dict={X: X_batch})

        reconstructions = outputs.eval(feed_dict={X: data[0: 10]})


    print(reconstructions.shape)

    plt.figure(figsize=(10, 4), dpi=100)
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(data[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()

        # display reconstruction
        ax = plt.subplot(2, 10, i + 10 + 1)
        plt.imshow(reconstructions[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()

    plt.show()