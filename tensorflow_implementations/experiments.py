import tensorflow as tf
from vanilla_autoencoder.vanilla_autoencoder import AUTOENCODER

from home.pn.PycharmProjects.autoencoders.helper_functions import *

if __name__ == '__main__':

    experimental_tasks1 = []
    settings = [(5, 0.0001), (10, 0.0001), (15, 0.0001)]
    train_data, test_data, m = get_mnist_data()

    for (num_epochs, l2_reg) in settings:

        model = AUTOENCODER()

        init = tf.global_variables_initializer()

        #Train
        train_loss, test_loss = [], []
        batch_size = 1500
        with tf.Session()   as sess:
            init.run()
            train_loss.append(model.loss.eval(session=sess, feed_dict={model.X: train_data}))
            test_loss.append(model.loss.eval(session=sess, feed_dict={model.X: test_data}))
            for epoch in range(num_epochs):
                print("Epoch:", epoch, "/", num_epochs)
                n_batches = m//batch_size
                for batch in range(n_batches):
                    X_batch = train_data[batch*batch_size: (batch + 1)*batch_size]
                    sess.run(model.training_op, feed_dict={model.X: X_batch})
                train_loss.append(model.loss.eval(session=sess, feed_dict={model.X: train_data}))
                test_loss.append(model.loss.eval(session=sess, feed_dict={model.X: test_data}))

            reconstructions = model.outputs.eval(feed_dict={model.X: test_data[0: 10]})

        experimental_tasks1.append(((num_epochs, l2_reg), train_loss, test_loss))

        plot_reconstructions(test_data[0: 10], reconstructions)

    plot_learning_curves([experimental_tasks1])
    #print(experimental_tasks1)