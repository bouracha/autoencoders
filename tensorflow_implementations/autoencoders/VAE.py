import tensorflow as tf
from functools import partial

class AUTOENCODER(object):

    def __init__(self, variational=False):
        n = 28 * 28  # for MNIST
        #Encoding Layers
        n_hidden1 = 500
        #Encoded Layer
        self.n_encoded = 5
        #Decoding Layers
        n_hidden3 = n_hidden1

        self.learning_rate = 0.001

        activation = tf.nn.elu
        initializer = tf.contrib.layers.variance_scaling_initializer()

    def build(self):
        self.X = tf.placeholder(tf.float32, shape=[None, n])

        self.reg = 0
        l = 1e-4

        #Initialise Weights Encoder
        weights1_init = initializer([n, n_hidden1])
        if variational:
            weights2_mu_init = initializer([n_hidden1, self.n_encoded])
            weights2_sigma_init = initializer([n_hidden1, self.n_encoded])
        else:
            weights2_init = initializer([n_hidden1, self.n_encoded])
        #Initialise Weights Decoder
        weights3_init = initializer([self.n_encoded, n_hidden3])
        weights4_init = initializer([n_hidden3, n])

        #Encoder Weights and Biases
        self.weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
        if variational:
            self.weights2_mu = tf.Variable(weights2_mu_init, dtype=tf.float32, name="weights2_mu")
            self.weights2_sigma = tf.Variable(weights2_sigma_init, dtype=tf.float32, name="weights2_sigma")
            self.biases2_mu = tf.Variable(tf.zeros(self.n_encoded), name="biases2_mu")
            self.biases2_sigma = tf.Variable(tf.zeros(self.n_encoded), name="biases2_sigma")
        else:
            self.weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
            self.biases2 = tf.Variable(tf.zeros(self.n_encoded), name="biases2")
        self.biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")

        #Decoder Weights and Biases
        self.weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
        self.weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")
        self.biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
        self.biases4 = tf.Variable(tf.zeros(n), name="biases4")

        #Regularisation Terms
        if variational:
            self.reg = tf.reduce_sum(tf.square(self.weights1)) \
                       + tf.reduce_sum(tf.square(self.weights2_mu)) \
                       + tf.reduce_sum(tf.square(self.weights2_sigma)) \
                       + tf.reduce_sum(tf.square(self.weights3)) \
                       + tf.reduce_sum(tf.square(self.weights4))
        else:
            self.reg = tf.reduce_sum(tf.square(self.weights1)) \
                     + tf.reduce_sum(tf.square(self.weights2))   \
                     + tf.reduce_sum(tf.square(self.weights3))   \
                     + tf.reduce_sum(tf.square(self.weights4))


        #Encoding Operations
        self.normalised_X = (self.X - tf.reduce_min(self.X))/(tf.reduce_max(self.X) - tf.reduce_min(self.X))
        self.variance_x = tf.reduce_mean(tf.square(self.normalised_X - tf.reduce_mean(self.normalised_X)))
        self.encoder_hidden1 = activation(tf.matmul(self.normalised_X, self.weights1) + self.biases1)
        #Encoded Layer
        if variational:
            self.encoded_mean = tf.matmul(self.encoder_hidden1, self.weights2_mu) + self.biases2_mu
            self.encoded_gamma = tf.matmul(self.encoder_hidden1, self.weights2_sigma) + self.biases2_sigma
            self.noise = tf.random_normal(tf.shape(self.encoded_gamma), dtype=tf.float32)
            self.encoded = self.encoded_mean + tf.exp(self.encoded_gamma) * self.noise
        else:
            self.encoded = tf.matmul(self.encoder_hidden1, self.weights2) + self.biases2
        #Decoding Operations
        self.decoder_hidden1 = activation(tf.matmul(self.encoded, self.weights3) + self.biases3)
        self.logits = tf.matmul(self.decoder_hidden1, self.weights4) + self.biases4
        self.outputs = tf.sigmoid(self.logits)

        #Loss Function
        #self.xentropy = tf.maximum(self.logits, 0) - tf.multiply(self.logits, self.normalised_X) + tf.log(1 + tf.exp(-tf.abs(self.logits)))
        self.xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.normalised_X, logits=self.logits)
        self.reconstruction_loss_xentropy = tf.reduce_mean(tf.reduce_sum(self.xentropy, axis=-1))
        self.reconstruction_loss_MSE = tf.reduce_mean(tf.square(self.logits - self.X))
        if variational:
            self.KL = 0.5 * tf.reduce_sum(tf.exp(self.encoded_gamma) + tf.square(self.encoded_mean) - 1 - self.encoded_gamma, axis=-1)
            self.loss = self.reconstruction_loss_xentropy + tf.reduce_mean(self.KL) + l*self.reg
        else:
            self.loss = self.reconstruction_loss_xentropy + l*self.reg

        #Optimiser
        self.optimizer = tf.train.AdamOptimizer(self.self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)