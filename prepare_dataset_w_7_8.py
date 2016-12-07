import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

logits_cumbersome = np.loadtxt('Data/logits_mnist_tuned2.csv', delimiter=', ')

batch_size = 50

# TRAINING SET
iters_per_epoch = mnist.train.num_examples/batch_size

train_xs = []
train_ys = []
train_logits = []
for i in range(iters_per_epoch):
    batch = mnist.train.next_batch(batch_size,shuffle=False)
    x_batch = batch[0]
    y_batch = batch[1]
    logits_batch = logits_cumbersome[i*batch_size:(i+1)*batch_size,:]

    rows_where_y_is_7_or_8 = np.any(np.equal(y_batch[:,[7,8]],np.ones([y_batch.shape[0],2])), axis=1)

    train_xs.append(x_batch[rows_where_y_is_7_or_8, :])
    train_ys.append(y_batch[rows_where_y_is_7_or_8, :])
    train_logits.append(logits_batch[rows_where_y_is_7_or_8, :])

train_xs = np.concatenate(train_xs, axis=0)
train_ys = np.concatenate(train_ys, axis=0)
train_logits = np.concatenate(train_logits, axis=0)

np.savetxt('Data/mnist_w_7_8_train_features.csv', train_xs, delimiter=', ')
np.savetxt('Data/mnist_w_7_8_train_labels.csv', train_ys, fmt='%d', delimiter=', ')
np.savetxt('Data/mnist_w_7_8_train_logits.csv', train_logits, delimiter=', ')
