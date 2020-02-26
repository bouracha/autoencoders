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
        n_neurons = [784, 500, self.n_encoded]
        #n_layers_rev = n_layers.copy().reverse()
        self.encode_layers = n_neurons
        self.decode_layers = [self.n_encoded, 500, 784] #n_layers_rev
        self.n_layers = len(n_neurons)

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
        n_encoder_weights = self.n_layers - 2
        for i in range(n_encoder_weights):
            weight_init = self.initializer([self.encode_layers[i], self.encode_layers[i+1]])
            self.encoder_weights["weights{0}".format(i)] = tf.Variable(weight_init, dtype=tf.float32, name="e_weights"+str(i))
            self.encoder_biases["biases{0}".format(i)] = tf.Variable(tf.zeros(self.encode_layers[i+1]), name="e_biases"+str(i))

        #Initialise Weights for Encoded layer -- depends on variational flag
        if self.variational:
            weights_mu_init = self.initializer([self.n_hidden1, self.n_encoded])
            weights_sigma_init = self.initializer([self.n_hidden1, self.n_encoded])
        else:
            weights_init = self.initializer([self.n_hidden1, self.n_encoded])
        if self.variational:
            self.weights_mu = tf.Variable(weights_mu_init, dtype=tf.float32, name="weights2_mu")
            self.weights_sigma = tf.Variable(weights_sigma_init, dtype=tf.float32, name="weights2_sigma")
            self.biases_mu = tf.Variable(tf.zeros(self.n_encoded), name="biases2_mu")
            self.biases_sigma = tf.Variable(tf.zeros(self.n_encoded), name="biases2_sigma")
        else:
            self.weights = tf.Variable(weights_init, dtype=tf.float32, name="weights2")
            self.biases = tf.Variable(tf.zeros(self.n_encoded), name="biases2")

        #Initialise Weights Decoder
        self.decoder_weights = {}
        self.decoder_biases = {}
        n_decoder_weights = self.n_layers - 1
        for i in range(n_decoder_weights):
            weight_init = self.initializer([self.decode_layers[i], self.decode_layers[i+1]])
            self.decoder_weights["weights{0}".format(i)] = tf.Variable(weight_init, dtype=tf.float32, name="d_weights"+str(i))
            self.decoder_biases["biases{0}".format(i)] = tf.Variable(tf.zeros(self.decode_layers[i+1]), name="d_biases"+str(i))

    def graph(self):
        #Regularisation Terms
        self.reg = 0

        self.encoder()
        self.decoder()


    def encoder(self):
        #Encoding Operations
        self.normalised_X = (self.X - tf.reduce_min(self.X))/(tf.reduce_max(self.X) - tf.reduce_min(self.X))
        self.variance_x = tf.reduce_mean(tf.square(self.normalised_X - tf.reduce_mean(self.normalised_X)))
        self.encoder_hiddens = {}
        layer_in = self.normalised_X
        n_layers = self.n_layers - 2
        for i in range(n_layers):
            self.encoder_hiddens["hiddens{0}".format(i)] = self.activation(tf.matmul(layer_in, self.encoder_weights["weights{0}".format(i)]) + self.encoder_biases["biases{0}".format(i)])
            self.reg += tf.reduce_sum(tf.square(self.encoder_weights["weights{0}".format(i)]))
            layer_in = self.encoder_hiddens["hiddens{0}".format(i)]

        #Encoded Layer
        if self.variational:
            self.encoded_mean = tf.matmul(self.encoder_hiddens["hiddens{0}".format(n_layers-1)], self.weights_mu) + self.biases_mu
            self.encoded_gamma = tf.matmul(self.encoder_hiddens["hiddens{0}".format(n_layers-1)], self.weights_sigma) + self.biases_sigma
            self.reg += tf.reduce_sum(tf.square(self.weights_mu)) \
                      + tf.reduce_sum(tf.square(self.weights_sigma))

            self.noise = tf.random_normal(tf.shape(self.encoded_gamma), dtype=tf.float32)
            self.encoded = self.encoded_mean + tf.exp(self.encoded_gamma) * self.noise
        else:
            self.encoded = tf.matmul(self.encoder_hiddens["hiddens{0}".format(n_layers-1)], self.weights) + self.biases
            self.reg += tf.reduce_sum(tf.square(self.weights))

    def decoder(self):
        # Decoding Operations
        self.decoder_hiddens = {}
        a_last = self.encoded
        n_activated_encoder_layers = self.n_layers-2
        for i in range(n_activated_encoder_layers):
            W = self.decoder_weights["weights{0}".format(i)]
            b = self.decoder_biases["biases{0}".format(i)]
            a = self.activation(tf.matmul(a_last, W) + b)
            self.decoder_hiddens["hiddens{0}".format(i)] = a
            self.reg += tf.reduce_sum(tf.square(W))
            a_last = a
        #Final unactivated decoding layer
        W = self.decoder_weights["weights{0}".format(n_activated_encoder_layers)]
        b = self.decoder_biases["biases{0}".format(n_activated_encoder_layers)]
        self.logits = tf.matmul(a_last, W) + b
        self.outputs = tf.sigmoid(self.logits)

