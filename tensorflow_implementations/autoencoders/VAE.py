import tensorflow as tf

class AUTOENCODER(object):

    def __init__(self, n, variational=False, learning_rate=0.001):
        self.n = n
        #Encoding Layers
        self.n_hidden1 = 500
        #Encoded Layer
        self.n_encoded = 200
        #Decoding Layers
        self.n_hidden3 = self.n_hidden1
        n_neurons = [784, 2000, 3000, self.n_encoded]
        #n_layers_rev = n_layers.copy().reverse()
        self.encode_layers = n_neurons
        self.decode_layers = [self.n_encoded, 3000, 2000, 784] #n_layers_rev
        self.n_layers = len(n_neurons)

        self.variational = variational
        self.learning_rate = learning_rate

        self.activation = tf.nn.elu
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        self.build()

    def build(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n])

        self.reg = 0

        self.initialize_variables()
        self.graph()
        self.losses()

        #Optimiser
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

    def initialize_variables(self):
        #Initialise Weights Encoder
        self.encoder_variables = {}
        n_encoder_weights = self.n_layers - 2
        for i in range(n_encoder_weights):
            weight_init = self.initializer([self.encode_layers[i], self.encode_layers[i+1]])
            self.encoder_variables["weights{0}".format(i)] = tf.Variable(weight_init, dtype=tf.float32, name="e_weights"+str(i))
            self.encoder_variables["biases{0}".format(i)] = tf.Variable(tf.zeros(self.encode_layers[i+1]), name="e_biases"+str(i))

        #Initialise Weights for Encoded layer -- depends on variational flag
        if self.variational:
            weights_mu_init = self.initializer([self.encode_layers[n_encoder_weights], self.n_encoded])
            weights_sigma_init = self.initializer([self.encode_layers[n_encoder_weights], self.n_encoded])
            self.weights_mu = tf.Variable(weights_mu_init, dtype=tf.float32, name="weights2_mu")
            self.weights_sigma = tf.Variable(weights_sigma_init, dtype=tf.float32, name="weights2_sigma")
            self.biases_mu = tf.Variable(tf.zeros(self.n_encoded), name="biases2_mu")
            self.biases_sigma = tf.Variable(tf.zeros(self.n_encoded), name="biases2_sigma")
        else:
            weights_init = self.initializer([self.encode_layers[n_encoder_weights], self.n_encoded])
            self.weights = tf.Variable(weights_init, dtype=tf.float32, name="weights2")
            self.biases = tf.Variable(tf.zeros(self.n_encoded), name="biases2")

        #Initialise Weights Decoder
        self.decoder_variables = {}
        n_decoder_weights = self.n_layers - 1
        for i in range(n_decoder_weights):
            weight_init = self.initializer([self.decode_layers[i], self.decode_layers[i+1]])
            self.decoder_variables["weights{0}".format(i)] = tf.Variable(weight_init, dtype=tf.float32, name="d_weights"+str(i))
            self.decoder_variables["biases{0}".format(i)] = tf.Variable(tf.zeros(self.decode_layers[i+1]), name="d_biases"+str(i))

    def graph(self):
        #Regularisation Terms

        self.encoder()
        self.decoder()


    def encoder(self):
        #Encoding Operations for dense activation layers
        a_last = self.X
        n_activated_encoder_layers = self.n_layers - 2

        self.encoder_hiddens = self.dense_layers(n_activated_encoder_layers, a_last, self.encoder_variables)

        #Encoded Layer
        a_last = self.encoder_hiddens["hiddens{0}".format(n_activated_encoder_layers-1)]
        if self.variational:
            z, mu, gamma = self.reparametisation_trick(a_last)
            self.encoded_mu = mu
            self.encoded_gamma = gamma
            self.encoded = z
        else:
            W = self.weights
            b = self.biases
            self.reg += tf.reduce_sum(tf.square(W))
            self.encoded = tf.matmul(a_last, W) + b

    def decoder(self):
        # Decoding Operations for dense activation layers
        a_last = self.encoded
        n_activated_decoder_layers = self.n_layers - 2

        self.decoder_hiddens = self.dense_layers(n_activated_decoder_layers, a_last, self.decoder_variables)

        #Final unactivated decoding layer
        W = self.decoder_variables["weights{0}".format(n_activated_decoder_layers)]
        b = self.decoder_variables["biases{0}".format(n_activated_decoder_layers)]
        self.reg += tf.reduce_sum(tf.square(W))
        a_last = self.decoder_hiddens["hiddens{0}".format(n_activated_decoder_layers-1)]
        self.logits = tf.matmul(a_last, W) + b
        self.outputs = tf.sigmoid(self.logits)

    def losses(self):
        #Loss Function
        l = 1e-4
        X_hat_sigmoid = self.outputs
        X_hat = self.logits
        X = self.X

        ### XEntropy
        #xentropy = tf.maximum(self.logits, 0) - tf.multiply(self.logits, self.X) + tf.log(1 + tf.exp(-tf.abs(self.logits)))
        xentropy_per_neuron_reconstruction = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=X_hat)
        summed_xentropy_per_example = tf.reduce_sum(xentropy_per_neuron_reconstruction, axis=-1)
        self.reconstruction_loss_xentropy = tf.reduce_mean(summed_xentropy_per_example)

        ### MSE
        self.reconstruction_loss_MSE = tf.reduce_mean(tf.square(X_hat_sigmoid - X))

        if self.variational:
            ### KL divergence for latent loss
            KL_per_example = self.KL()
            self.latent_loss = tf.reduce_mean(KL_per_example)
            self.loss = self.reconstruction_loss_xentropy + self.latent_loss + l*self.reg
        else:
            self.loss = self.reconstruction_loss_xentropy + l*self.reg

    # Methods
    def dense_layers(self, n_activated_layers, a_last, variables):
        hiddens = {}
        for i in range(n_activated_layers):
            W = variables["weights{0}".format(i)]
            b = variables["biases{0}".format(i)]
            self.reg += tf.reduce_sum(tf.square(W))

            a = self.activation(tf.matmul(a_last, W) + b)

            hiddens["hiddens{0}".format(i)] = a
            a_last = a
        return hiddens

    def reparametisation_trick(self, a_last):
        W_mu = self.weights_mu
        b_mu = self.biases_mu
        W_sigma = self.weights_sigma
        b_sigma = self.biases_sigma
        self.reg += tf.reduce_sum(tf.square(W_mu)) \
                    + tf.reduce_sum(tf.square(W_sigma))

        mu = tf.matmul(a_last, W_mu) + b_mu
        gamma = tf.matmul(a_last, W_sigma) + b_sigma

        noise = tf.random_normal(tf.shape(gamma), dtype=tf.float32)

        z = mu + tf.exp(gamma) * noise

        return z, mu, gamma

    def KL(self):
        KL = 0.5 * tf.reduce_sum(tf.exp(self.encoded_gamma) + tf.square(self.encoded_mu) - 1 - self.encoded_gamma, axis=-1)
        return KL
