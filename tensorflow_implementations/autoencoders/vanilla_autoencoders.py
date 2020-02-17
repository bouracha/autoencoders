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


class denoising_AUTOENCODER_300_150_300(object):

    def __init__(self, l2_reg=0.0001, noise_level=10.0):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01

        self.X = tf.placeholder(tf.float32, shape=[None, n])
        self.X_noisey = self.X + noise_level*tf.random_normal(tf.shape(self.X))

        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        #self.l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        self.my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=self.he_init)

        self.hidden1 = self.my_dense_layer(self.X_noisey, 300)
        self.hidden2 = self.my_dense_layer(self.hidden1, 150)
        self.hidden3 = self.my_dense_layer(self.hidden2, 300)
        self.outputs = self.my_dense_layer(self.hidden3, n, activation=None)  ##Overwrite: no activation fn in last layer

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X_noisey))  # MSE

        #self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = self.reconstruction_loss#tf.add_n([self.reconstruction_loss] + self.reg_losses)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)


class sparse_AUTOENCODER_300_150_300(object):

    def __init__(self, sparsity_target=0.1, sparsity_weight=0.2):
        n = 28 * 28  # for MNIST

        learning_rate = 0.01

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        #self.l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ## Partial allows to use the function my_dense_layer with same set parameters each time
        self.my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=self.he_init)

        self.hidden1 = self.my_dense_layer(self.X, 300) ##So the activations in the coding layer must be between 0 and 1
        self.hidden2 = self.my_dense_layer(self.hidden1, 150)
        self.hidden3 = self.my_dense_layer(self.hidden2, 300)
        self.outputs = self.my_dense_layer(self.hidden3, n, activation=None)  ##Overwrite: no activation fn in last layer

        #Sparsity loss calculation
        self.sparsity_target = sparsity_target
        self.hidden1_mean = tf.reduce_mean(self.hidden1, axis=0)  # batch mean for particular neuron
        self.hidden2_mean = tf.reduce_mean(self.hidden2, axis=0)  # batch mean for particular neuron
        self.sparcity_loss1 = tf.reduce_sum(self.kl_divergence(self.hidden1_mean))
        self.sparcity_loss2 = tf.reduce_sum(self.kl_divergence(self.hidden2_mean))

        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))  # MSE

        #self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = self.reconstruction_loss + sparsity_weight*self.sparcity_loss1 + sparsity_weight*self.sparcity_loss2

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

    def kl_divergence(self, q):
        p = self.sparsity_target
        return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))



class AUTOENCODER_500_500_20(object):

    def __init__(self):
        n = 28 * 28  # for MNIST
        #Encoding Layers
        n_hidden1 = 500
        n_hidden2 = 500
        #Encoded Layer
        self.n_hidden3 = 20
        #Decoding Layers
        n_hidden4 = n_hidden2
        n_hidden5 = n_hidden1

        learning_rate = 0.001

        activation = tf.nn.elu
        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        #Initialise Weights Encoder
        weights1_init = initializer([n, n_hidden1])
        weights2_init = initializer([n_hidden1, n_hidden2])
        weights3_init = initializer([n_hidden2, self.n_hidden3])
        #Initialise Weights Decoder
        weights4_init = initializer([self.n_hidden3, n_hidden4])
        weights5_init = initializer([n_hidden4, n_hidden5])
        weights6_init = initializer([n_hidden5, n])

        #Encoder Weights and Biases
        self.weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
        self.weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
        self.weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
        self.biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
        self.biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
        self.biases3 = tf.Variable(tf.zeros(self.n_hidden3), name="biases3")

        #Decoder Weights and Biases
        self.weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")
        self.weights5 = tf.Variable(weights5_init, dtype=tf.float32, name="weights5")
        self.weights6 = tf.Variable(weights6_init, dtype=tf.float32, name="weights6")
        self.biases4 = tf.Variable(tf.zeros(n_hidden4), name="biases4")
        self.biases5 = tf.Variable(tf.zeros(n_hidden5), name="biases5")
        self.biases6 = tf.Variable(tf.zeros(n), name="biases6")

        #Encoding Operations
        self.normalised_X = (self.X - tf.reduce_min(self.X))/(tf.reduce_max(self.X) - tf.reduce_min(self.X))
        self.encoder_hidden1 = activation(tf.matmul(self.normalised_X, self.weights1) + self.biases1)
        self.encoder_hidden2 = activation(tf.matmul(self.encoder_hidden1, self.weights2) + self.biases2)
        self.encoded = tf.matmul(self.encoder_hidden2, self.weights3) + self.biases3
        #Decoding Operations
        self.decoder_hidden1 = activation(tf.matmul(self.encoded, self.weights4) + self.biases4)
        self.decoder_hidden2 = activation(tf.matmul(self.decoder_hidden1, self.weights5) + self.biases5)
        self.logits = tf.matmul(self.decoder_hidden2, self.weights6) + self.biases6
        self.outputs = self.logits

        #Loss Function
        self.xentropy = tf.maximum(self.logits, 0) - tf.multiply(self.logits, self.normalised_X) + tf.log(1 + tf.exp(-tf.abs(self.logits)))
        self.reconstruction_loss_xentropy = tf.reduce_mean(self.xentropy)
        self.reconstruction_loss_MSE = tf.reduce_mean(tf.square(self.logits - self.X))
        self.loss = self.reconstruction_loss_xentropy

        #Optimiser
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)


class VARIATIONAL_AUTOENCODER_500_500_20(object):

    def __init__(self):
        n = 28 * 28  # for MNIST
        #Encoding Layers
        n_hidden1 = 500
        n_hidden2 = 500
        #Encoded Layer
        self.n_hidden3 = 20
        #Decoding Layers
        n_hidden4 = n_hidden2
        n_hidden5 = n_hidden1

        learning_rate = 0.001

        activation = tf.nn.elu
        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.X = tf.placeholder(tf.float32, shape=[None, n])

        #Initialise Weights Encoder
        weights1_init = initializer([n, n_hidden1])
        weights2_init = initializer([n_hidden1, n_hidden2])
        weights3_init = initializer([n_hidden2, self.n_hidden3])
        #Initialise Weights Decoder
        weights4_init = initializer([self.n_hidden3, n_hidden4])
        weights5_init = initializer([n_hidden4, n_hidden5])
        weights6_init = initializer([n_hidden5, n])

        #Encoder Weights and Biases
        self.weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
        self.weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
        self.weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
        self.biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
        self.biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
        self.biases3 = tf.Variable(tf.zeros(self.n_hidden3), name="biases3")

        #Decoder Weights and Biases
        self.weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")
        self.weights5 = tf.Variable(weights5_init, dtype=tf.float32, name="weights5")
        self.weights6 = tf.Variable(weights6_init, dtype=tf.float32, name="weights6")
        self.biases4 = tf.Variable(tf.zeros(n_hidden4), name="biases4")
        self.biases5 = tf.Variable(tf.zeros(n_hidden5), name="biases5")
        self.biases6 = tf.Variable(tf.zeros(n), name="biases6")

        #Encoding Operations
        self.normalised_X = (self.X - tf.reduce_min(self.X))/(tf.reduce_max(self.X) - tf.reduce_min(self.X))
        self.encoder_hidden1 = activation(tf.matmul(self.normalised_X, self.weights1) + self.biases1)
        self.encoder_hidden2 = activation(tf.matmul(self.encoder_hidden1, self.weights2) + self.biases2)
        #Encoded Layer
        self.encoded_mean = tf.matmul(self.encoder_hidden2, self.weights3) + self.biases3
        self.encoded_gamma = tf.matmul(self.encoder_hidden2, self.weights3) + self.biases3
        self.noise = tf.random_normal(tf.shape(self.encoded_gamma), dtype=tf.float32)
        self.encoded = self.encoded_mean + tf.exp(0.5*self.encoded_gamma)*self.noise
        #Decoding Operations
        self.decoder_hidden1 = activation(tf.matmul(self.encoded, self.weights4) + self.biases4)
        self.decoder_hidden2 = activation(tf.matmul(self.decoder_hidden1, self.weights5) + self.biases5)
        self.logits = tf.matmul(self.decoder_hidden2, self.weights6) + self.biases6
        self.outputs = self.logits

        #Loss Function
        self.xentropy = tf.maximum(self.logits, 0) - tf.multiply(self.logits, self.normalised_X) + tf.log(1 + tf.exp(-tf.abs(self.logits)))
        self.reconstruction_loss_xentropy = tf.reduce_mean(self.xentropy)
        self.reconstruction_loss_MSE = tf.reduce_mean(tf.square(self.logits - self.X))
        self.latent_loss = 0.5*tf.reduce_mean(tf.exp(self.encoded_gamma) + tf.square(self.encoded_mean) - 1 - self.encoded_gamma)
        self.loss = self.reconstruction_loss_MSE + self.latent_loss

        #Optimiser
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)


