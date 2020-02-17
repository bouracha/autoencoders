import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from autoencoders.vanilla_autoencoders import AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import AUTOENCODER_150
from autoencoders.vanilla_autoencoders import AUTOENCODER_50
from autoencoders.vanilla_autoencoders import tied_AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import denoising_AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import sparse_AUTOENCODER_300_150_300
from autoencoders.vanilla_autoencoders import VARIATIONAL_AUTOENCODER_500_500_20

from autoencoders.vanilla_autoencoders import VARIATIONAL_AUTOENCODER_500_500_20_original

from home.pn.PycharmProjects.autoencoders.helper_functions import *

import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_data, test_data, m = get_mnist_data()

    num_epochs = 50

    model = VARIATIONAL_AUTOENCODER_500_500_20()

    init = tf.global_variables_initializer()

    #Train
    train_loss, test_loss = [], []
    batch_size = 200
    final_losses = []
    with tf.Session()   as sess:
        init.run()
        train_loss.append(model.loss.eval(session=sess, feed_dict={model.X: train_data}))
        test_loss.append(model.loss.eval(session=sess, feed_dict={model.X: test_data}))
        print("Number of Epochs = " + str(num_epochs))
        for epoch in range(num_epochs):
            print(str(epoch) + "/" + str(num_epochs), end="\r")
            n_batches = m//batch_size
            for batch in range(n_batches):
                X_batch = train_data[batch*batch_size: (batch + 1)*batch_size]
                sess.run(model.training_op, feed_dict={model.X: X_batch})
            train_loss.append(model.loss.eval(session=sess, feed_dict={model.X: train_data}))
            test_loss.append(model.loss.eval(session=sess, feed_dict={model.X: test_data}))

        print("Train Loss: ", train_loss[-1])
        print("Test Loss: ", test_loss[-1])
        reconstructions = model.outputs.eval(feed_dict={model.X: test_data[0: 10]})
        xentropy_loss = model.reconstruction_loss_xentropy.eval(session=sess, feed_dict={model.X: test_data})
        MSE_loss = model.reconstruction_loss_MSE.eval(session=sess, feed_dict={model.X: test_data})
        print("XEntropy Loss: ", xentropy_loss)
        print("MSE Loss: ", MSE_loss)

        codings_rnd = np.random.normal(size=[60, model.n_hidden3])
        outputs_val = model.outputs.eval(feed_dict={model.encoded: codings_rnd})

    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title(' Learning Curves')
    plt.legend()

    plot_reconstructions(test_data[0: 10], reconstructions)
    print(test_loss)


    for iteration in range(60):
        plt.subplot(60,10,iteration+1)
        plot_image(outputs_val[iteration])