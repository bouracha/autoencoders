import tensorflow as tf
from functools import partial

class AUTOENCODER_300_150_300(object):

    def __init__(self, l2_reg):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        #self.l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        self.my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=self.he_init)

        self.hidden1 = self.my_dense_layer(self.X, 300)
        self.hidden2 = self.my_dense_layer(self.hidden1, 150)
        self.hidden3 = self.my_dense_layer(self.hidden2, 300)
        self.outputs = self.my_dense_layer(self.hidden3, n, activation=None)  ##Overwrite: no activation fn in last layer

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))  # MSE

        #self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = self.reconstruction_loss#tf.add_n([self.reconstruction_loss] + self.reg_losses)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)


class AUTOENCODER_150(object):

    def __init__(self, l2_reg):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        self.my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=self.he_init, kernel_regularizer=self.l2_regularizer)

        self.hidden = self.my_dense_layer(self.X, 150)
        self.outputs = self.my_dense_layer(self.hidden, n)  ##Overwrite: no activation fn in last layer

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))  # MSE

        self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)


class AUTOENCODER_50(object):

    def __init__(self, l2_reg):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        self.my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=self.he_init, kernel_regularizer=self.l2_regularizer)

        self.hidden = self.my_dense_layer(self.X, 50)
        self.outputs = self.my_dense_layer(self.hidden, n)  ##Overwrite: no activation fn in last layer

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))  # MSE

        self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

class tied_AUTOENCODER_300_150_300(object):

    def __init__(self, l2_reg):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01

        activation = tf.nn.elu
        initializer = tf.contrib.layers.variance_scaling_initializer()
        l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        weights1_init = initializer([n, 300])
        weights2_init = initializer([300, 150])

        self.weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
        self.weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
        self.weights3 = tf.transpose(self.weights2, name="weights3")  # tied weights
        self.weights4 = tf.transpose(self.weights1, name="weights4")# tied weights

        self.biases1 = tf.Variable(tf.zeros(300), name="biases1")
        self.biases2 = tf.Variable(tf.zeros(150), name="biases2")
        self.biases3 = tf.Variable(tf.zeros(300), name="biases3")
        self.biases4 = tf.Variable(tf.zeros(n), name="biases4")

        self.hidden1 = activation(tf.matmul(self.X, self.weights1) + self.biases1)
        self.hidden2 = activation(tf.matmul(self.hidden1, self.weights2) + self.biases2)
        self.hidden3 = activation(tf.matmul(self.hidden2, self.weights3) + self.biases3)
        self.outputs = tf.matmul(self.hidden3, self.weights4) + self.biases4

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))  # MSE
        self.reg_losses = l2_regularizer(self.weights1) + l2_regularizer(self.weights2)
        self.loss = self.reconstruction_loss + self.reg_losses

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)