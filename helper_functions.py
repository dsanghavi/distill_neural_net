import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def accuracy_in_batches(data, accuracy_func, x, y_, keep_prob=None, batch_size=50):
    y_accs = []
    for j in range(data.num_examples/batch_size):
        check_batch = data.next_batch(batch_size)
        if keep_prob is None:
            y_accs.append(accuracy_func.eval(feed_dict={
                    x: check_batch[0], y_: check_batch[1]}))
        else:
            y_accs.append(accuracy_func.eval(feed_dict={
                    x: check_batch[0], y_: check_batch[1], keep_prob: 1.0}))
    return np.mean(y_accs)

def accuracy_in_batches_alt(features, labels, accuracy_func, x, y_, keep_prob=None, batch_size=50):
    y_accs = []
    iters_per_epoch = np.ceil(float(features.shape[0]) / batch_size).astype(int)
    for i in range(iters_per_epoch):
        start_index = (i % iters_per_epoch) * batch_size
        end_index = ((i % iters_per_epoch) + 1) * batch_size
        if end_index > features.shape[0]:
            end_index = features.shape[0]

        features_batch = features[start_index:end_index]
        labels_batch = labels[start_index:end_index]

        if keep_prob is None:
            y_accs.append(accuracy_func.eval(feed_dict={x: features_batch, y_: labels_batch}))
        else:
            y_accs.append(accuracy_func.eval(feed_dict={x: features_batch, y_: labels_batch, keep_prob: 1.0}))
    return np.mean(y_accs)

def softmax_T(logits, T, tensor=False):
    # train_logits dim = (N, C), T = float
    if tensor:
        l1 = tf.minimum(tf.exp(logits / T), 1e12)
        l1 /= tf.maximum(tf.reduce_sum(l1, reduction_indices=1, keep_dims=True), 1e-12)
    else:
        l1 = np.minimum(np.exp(logits / T), 1e12)
        l1 /= np.maximum(np.sum(l1, axis=1, keepdims=True), 1e-12)
    return l1
