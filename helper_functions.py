import numpy as np
import tensorflow as tf

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

def softmax_T(logits, T, tensor=False):
    # logits dim = (N, C), T = float
    if tensor:
        l1 = tf.exp(logits / T)
        l1 = l1 / tf.reduce_sum(l1, reduction_indices=1)[:, tf.newaxis]
    else:
        l1 = np.exp(logits / T)
        l1 = l1 / np.sum(l1, axis=1)[:, np.newaxis]
    return l1
