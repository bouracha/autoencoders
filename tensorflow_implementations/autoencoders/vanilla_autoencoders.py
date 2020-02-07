import tensorflow as tf
from functools import partial

class AUTOENCODER(object):

    def __init__(self):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01
        l2_reg = 0.0001

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        self.my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=self.he_init, kernel_regularizer=self.l2_regularizer)

        self.hidden1 = self.my_dense_layer(self.X, 300)
        self.hidden2 = self.my_dense_layer(self.hidden1, 150)
        self.hidden3 = self.my_dense_layer(self.hidden2, 300)
        self.outputs = self.my_dense_layer(self.hidden3, n, activation=None)  ##Overwrite: no activation fn in last layer

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))  # MSE

        self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)
