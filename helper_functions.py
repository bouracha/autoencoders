import hashlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

import pandas as pd

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def split_train_crossvalidate(X, y, cv_ratio):
    shuffled_indices = np.random.permutation(len(X))
    cv_set_size = int(len(X) * cv_ratio)
    cv_indices = shuffled_indices[:cv_set_size]
    train_indices = shuffled_indices[cv_set_size:]

    return X.iloc[train_indices], y.iloc[train_indices], X.iloc[cv_indices], y.iloc[cv_indices]


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


def y_to_onehot(y):
    y = np.array(y)
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    return y

##############
#DeepLearning
##############

# Global variables.
log_period_samples = 20000
batch_size = 100

def get_mnist_data():
    print("Reading MNIST data from file..")

    data = np.array(pd.read_csv('mnist_data/data.csv'))
    labels = np.array(pd.read_csv('mnist_data/target.csv'))

    print('Randomising train and test set...')
    np.random.seed(69)
    train_indices = np.random.permutation(70000)

    test_data = data[train_indices[:70000 // 5]]
    train_data = data[train_indices[70000 // 5:]]

    m = train_data.shape[0]

    print("Retrieved randomised MNIST data")
    return train_data, test_data, m

def plot_images(top_row, bottom_row):
    """ Arguments: list of 10 28x28 grayscale images, originals"""
    """            the reconstructions same as above """
    plt.figure(figsize=(20, 8), dpi=100)
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(top_row[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()
        # display reconstruction
        ax = plt.subplot(2, 10, i + 10 + 1)
        plt.imshow(bottom_row[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()
    plt.draw()

# Import dataset with one-hot encoding of the class labels.
def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.
def get_placeholders():
  x = tf.placeholder(tf.float32, [None, 114])
  y_ = tf.placeholder(tf.float32, [None, 6])
  return x, y_

# Plot learning curves of experiments
def plot_learning_curves(experiment_data):
  # Generate figure.
  fig, axes = plt.subplots(3, 3, figsize=(16,12))
  st = fig.suptitle(
      "Learning Curves for all Tasks and Hyper-parameter settings",
      fontsize="x-large")
  # Plot all learning curves.
  for i, results in enumerate(experiment_data):
    for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
      # Plot.
      xs = [x * log_period_samples for x in range(1, len(train_accuracy)+1)]
      axes[j, i].plot(xs, train_accuracy, label='train_accuracy')
      axes[j, i].plot(xs, test_accuracy, label='test_accuracy')
      # Prettify individual plots.
      axes[j, i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      axes[j, i].set_xlabel('Number of samples processed')
      axes[j, i].set_ylabel('Epochs: {}, Learning rate: {}.  Accuracy'.format(*setting))
      axes[j, i].set_title('Task {}'.format(i + 1))
      axes[j, i].legend()
  # Prettify overall figure.
  plt.tight_layout()
  st.set_y(0.95)
  fig.subplots_adjust(top=0.91)
  plt.show()

# Generate summary table of results.
def plot_summary_table(experiment_data):
  # Fill Data.
  cell_text = []
  rows = []
  columns = ['Setting 1', 'Setting 2', 'Setting 3']
  for i, results in enumerate(experiment_data):
    rows.append('Model {}'.format(i + 1))
    cell_text.append([])
    for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
      if test_accuracy != []:
        cell_text[i].append(test_accuracy[-1])
      else:
        print('Warning: Something went wrong! Missing testing/training data')
  # Generate Table.
  fig=plt.figure(frameon=False)
  ax = plt.gca()
  the_table = ax.table(
      cellText=cell_text,
      rowLabels=rows,
      colLabels=columns,
      loc='center')
  the_table.scale(1, 4)
  # Prettify.
  ax.patch.set_facecolor('None')
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)