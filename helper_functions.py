import numpy as np
import tensorflow as tf
import sys

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
        l1 = tf.exp(logits / T)
        l1 /= tf.maximum(tf.reduce_sum(l1, reduction_indices=1, keep_dims=True), 1e-12)
    else:
        l1 = np.exp(logits / T)
        l1 /= np.maximum(np.sum(l1, axis=1, keepdims=True), 1e-12)
    return l1

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, operation=tf.matmul):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # Initialization
        with tf.name_scope('weights'):
            if operation is conv2d:
                weights = weight_variable(input_dim)
            else:
                weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        # Operation
        with tf.name_scope('Wx_plus_b'):
            preactivate = operation(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        # Non linearity
        activations = act(preactivate, name='activation')
        tf.histogram_summary(layer_name + '/activations', activations)

        return activations, weights, biases


class Logger(object):
    """
    Logger class to print stdout messages into a log file while displaying them in stdout also.
    Pass filename to which to save when instantiating.
    """
    def __init__(self, f_log_name):
        """
        Initialization.
        :param f_log_name: file path to which to write the logs
        """
        self.terminal = sys.stdout
        self.log = open(f_log_name, 'w')

    def write(self, message):
        """
        Actual logging operations.
        :param message: The message to log.
        :return: Nothing
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Exists only to satisfy Python 3
        :return: Nothing
        """
        pass

    def close_log(self):
        """
        Closes the opened log file.
        :return:
        """
        self.log.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exists to close the log file in case user terminated the script, etc., and close_log is not explicitly called.
        """
        self.close_log()

