import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from autoencoders.vanilla_autoencoders import AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import tied_AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import denoising_AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import sparse_AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import AUTOENCODER_500_500_20

from autoencoders.VAE import AUTOENCODER

import time
import sys
sys.path.append("../")
from helper_functions import *
from IPython.display import clear_output

import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':

    train_data, test_data, m = get_mnist_data()

    #Normalise
    max = np.max(train_data)
    min = np.min(train_data)
    train_data = (train_data - min)/(max - min)
    test_data = (test_data - min)/(max - min)

    num_epochs = 10

    model = AUTOENCODER(variational=True, learning_rate=0.001, layers=[784, 500, 5])

    init = tf.global_variables_initializer()

    #Train
    train_loss, test_loss = [], []
    batch_size = 100
    final_losses = []
    #plt.ion()
    with tf.Session() as sess:
        init.run()
        train_loss.append(model.loss.eval(session=sess, feed_dict={model.X: train_data}))
        test_loss.append(model.loss.eval(session=sess, feed_dict={model.X: test_data}))
        print("Number of Epochs = " + str(num_epochs))
        start_time = time.time()
        for epoch in range(num_epochs):
            print(str(epoch) + "/" + str(num_epochs), end="\r")
            n_batches = m//batch_size
            for batch in range(n_batches):
                X_batch = train_data[batch*batch_size: (batch + 1)*batch_size]
                sess.run(model.training_op, feed_dict={model.X: X_batch})
            train_loss.append(model.loss.eval(session=sess, feed_dict={model.X: train_data}))
            test_loss.append(model.loss.eval(session=sess, feed_dict={model.X: test_data}))
        stop_time = time.time()
        print("Runtime = ", stop_time-start_time, "Seconds")

        print("Train Loss: ", train_loss[-1])
        print("Test Loss: ", test_loss[-1])
        reconstructions = model.outputs.eval(feed_dict={model.X: test_data[0: 10]})
        xentropy_loss = model.reconstruction_loss_xentropy.eval(session=sess, feed_dict={model.X: test_data})
        MSE_loss = model.reconstruction_loss_MSE.eval(session=sess, feed_dict={model.X: test_data})
        print("XEntropy Loss: ", xentropy_loss/784.0)
        print("MSE Loss: ", MSE_loss)

        codings_rnd = np.random.normal(size=[60, model.n_encoded])
        outputs_val = model.outputs.eval(feed_dict={model.encoded: codings_rnd})

    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='CV Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title(' Learning Curves')
    plt.legend()


    print('Train Loss list: ', train_loss)
    print('Test loss list', test_loss)

    plot_images(test_data[0: 10], reconstructions)
    plot_images(outputs_val[:10], outputs_val[10:20])


    plt.show()
