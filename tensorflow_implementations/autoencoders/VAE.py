import tensorflow as tf
from functools import partial

class AUTOENCODER(object):

    def __init__(self, variational=False, learning_rate=0.001):
        self.n = 28 * 28  # for MNIST
        #Encoding Layers
        self.n_hidden1 = 500
        #Encoded Layer
        self.n_encoded = 20
        #Decoding Layers
        self.n_hidden3 = self.n_hidden1
        n_layers = [784, 500, self.n_encoded]
        n_layers_rev = n_layers.copy().reverse()
        self.encode_layers = n_layers
        self.decode_layers = [self.n_encoded, 500, 784] #n_layers_rev

        self.variational = variational
        self.learning_rate = learning_rate

        self.activation = tf.nn.elu
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        self.build()

    def build(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n])

        self.reg = 0
        l = 1e-4

        self.initialize_variables()

        self.graph()

        #Loss Function
        #self.xentropy = tf.maximum(self.logits, 0) - tf.multiply(self.logits, self.normalised_X) + tf.log(1 + tf.exp(-tf.abs(self.logits)))
        self.xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.normalised_X, logits=self.logits)
        self.reconstruction_loss_xentropy = tf.reduce_mean(tf.reduce_sum(self.xentropy, axis=-1))
        self.reconstruction_loss_MSE = tf.reduce_mean(tf.square(self.logits - self.X))
        if self.variational:
            self.KL = 0.5 * tf.reduce_sum(tf.exp(self.encoded_gamma) + tf.square(self.encoded_mean) - 1 - self.encoded_gamma, axis=-1)
            self.loss = self.reconstruction_loss_xentropy + tf.reduce_mean(self.KL) + l*self.reg
        else:
            self.loss = self.reconstruction_loss_xentropy + l*self.reg

        #Optimiser
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

    def initialize_variables(self):
        #Initialise Weights Encoder
        self.encoder_weights = {}
        self.encoder_biases = {}
        for i in range(1):
            weight_init = self.initializer([self.encode_layers[i], self.encode_layers[i+1]])
            self.encoder_weights["weights{0}".format(i)] = tf.Variable(weight_init, dtype=tf.float32, name="e_weights"+str(i))
            self.encoder_biases["biases{0}".format(i)] = tf.Variable(tf.zeros(self.encode_layers[i+1]), name="e_biases"+str(i))
        if self.variational:
            weights2_mu_init = self.initializer([self.n_hidden1, self.n_encoded])
            weights2_sigma_init = self.initializer([self.n_hidden1, self.n_encoded])
        else:
            weights2_init = self.initializer([self.n_hidden1, self.n_encoded])
        #Initialise Weights Decoder
        self.dencoder_weights = {}
        for i in range(2):
            weight_init = self.initializer([self.decode_layers[i], self.decode_layers[i+1]])
            self.dencoder_weights["weights{0}".format(i)] = tf.Variable(weight_init, dtype=tf.float32, name="d_weights"+str(i))

        #Encoder Weights and Biases

        if self.variational:
            self.weights2_mu = tf.Variable(weights2_mu_init, dtype=tf.float32, name="weights2_mu")
            self.weights2_sigma = tf.Variable(weights2_sigma_init, dtype=tf.float32, name="weights2_sigma")
            self.biases2_mu = tf.Variable(tf.zeros(self.n_encoded), name="biases2_mu")
            self.biases2_sigma = tf.Variable(tf.zeros(self.n_encoded), name="biases2_sigma")
        else:
            self.weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
            self.biases2 = tf.Variable(tf.zeros(self.n_encoded), name="biases2")

        #Decoder Weights and Biases
        #self.weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
        #self.weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")
        self.biases3 = tf.Variable(tf.zeros(self.n_hidden3), name="biases3")
        self.biases4 = tf.Variable(tf.zeros(self.n), name="biases4")

    def graph(self):
        #Regularisation Terms
        if self.variational:
            self.reg = tf.reduce_sum(tf.square(self.encoder_weights['weights0'])) \
                       + tf.reduce_sum(tf.square(self.weights2_mu)) \
                       + tf.reduce_sum(tf.square(self.weights2_sigma)) \
                       + tf.reduce_sum(tf.square(self.dencoder_weights['weights0'])) \
                       + tf.reduce_sum(tf.square(self.dencoder_weights['weights1']))
        else:
            self.reg = tf.reduce_sum(tf.square(self.encoder_weights['weights0'])) \
                     + tf.reduce_sum(tf.square(self.weights2))   \
                     + tf.reduce_sum(tf.square(self.dencoder_weights['weights0']))   \
                     + tf.reduce_sum(tf.square(self.dencoder_weights['weights1']))

        self.encoder()
        self.decoder()


    def encoder(self):
        #Encoding Operations
        self.normalised_X = (self.X - tf.reduce_min(self.X))/(tf.reduce_max(self.X) - tf.reduce_min(self.X))
        self.variance_x = tf.reduce_mean(tf.square(self.normalised_X - tf.reduce_mean(self.normalised_X)))
        self.encoder_hidden1 = self.activation(tf.matmul(self.normalised_X, self.encoder_weights['weights0']) + self.encoder_biases['biases0'])
        #Encoded Layer
        if self.variational:
            self.encoded_mean = tf.matmul(self.encoder_hidden1, self.weights2_mu) + self.biases2_mu
            self.encoded_gamma = tf.matmul(self.encoder_hidden1, self.weights2_sigma) + self.biases2_sigma
            self.noise = tf.random_normal(tf.shape(self.encoded_gamma), dtype=tf.float32)
            self.encoded = self.encoded_mean + tf.exp(self.encoded_gamma) * self.noise
        else:
            self.encoded = tf.matmul(self.encoder_hidden1, self.weights2) + self.biases2

    def decoder(self):
        #Decoding Operations
        self.decoder_hidden1 = self.activation(tf.matmul(self.encoded, self.dencoder_weights['weights0']) + self.biases3)
        self.logits = tf.matmul(self.decoder_hidden1, self.dencoder_weights['weights1']) + self.biases4
        self.outputs = tf.sigmoid(self.logits)

