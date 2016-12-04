import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

logits_cumbersome = np.loadtxt('Data/logits_mnist1.txt')

batch_size = 50
iters_per_epoch = mnist.train.num_examples/batch_size

xs = []
ys = []
logits = []
for i in range(iters_per_epoch):
    batch = mnist.train.next_batch(batch_size,shuffle=False)
    x_batch = batch[0]
    y_batch = batch[1]
    logits_batch = logits_cumbersome[i*batch_size:(i+1)*batch_size,:]

    rows_where_y_is_not_3 = (y_batch[:,3]==0)

    xs.append(x_batch[rows_where_y_is_not_3, :])
    ys.append(y_batch[rows_where_y_is_not_3, :])
    logits.append(logits_batch[rows_where_y_is_not_3, :])

xs = np.concatenate(xs,axis=0)
ys = np.concatenate(ys,axis=0)
logits = np.concatenate(logits,axis=0)

np.savetxt('Data/mnist_wo_3_features.csv', xs, delimiter=', ')
np.savetxt('Data/mnist_wo_3_labels.csv', ys, fmt='%d', delimiter=', ')
np.savetxt('Data/mnist_wo_3_logits.csv', logits, delimiter=', ')
