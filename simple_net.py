import tensorflow as tf
import numpy as np
from helper_functions import *


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Read logits
logits_cumbersome = np.loadtxt('logits_mnist1.txt')


# train small neural net ----------------------------------------------------------

#Read input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# RUN PARAMETERS
Ts = np.logspace(0,np.log10(30),11)  # range should be fine, increase intervals when ready
alphas = np.linspace(0,1,6)         # range IS fine, increase intervals when ready
num_repeat = 5                      # could be 10
batch_size = 50                     # no need to change I guess
num_epochs = 2                      # could be changed... 5? 10? 20? write_logits.py uses 20.
hidden_sizes = [800, 800]           # could be increased when ready...

results_np = np.zeros([Ts.shape[0], alphas.shape[0]])
results_np_w_repeat = np.zeros([Ts.shape[0], alphas.shape[0], num_repeat])

# Tensorflow stuff
T_index = -1
for T in Ts:
    T_index += 1
    alpha_index = -1
    y_soft = softmax_T(logits_cumbersome,T)     # temperature softmax
    for alpha in alphas:
        alpha_index += 1
        # y_new = (alpha*y_soft) + ((1-alpha) * mnist.train.labels) # interpolate cross entropies, not y!!
        # assert np.mean(np.argmax(y_soft,axis=1) == np.argmax(mnist.train.labels,axis=1)) > 0.99
        avg_acc = 0
        for repeat in range(num_repeat):
            tf.reset_default_graph()

            sess = tf.InteractiveSession()

            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_soft_ = tf.placeholder(tf.float32, shape=[None, 10])
            y_hard_ = tf.placeholder(tf.float32, shape=[None, 10])

            #Layers
            W1 = weight_variable([28 * 28, hidden_sizes[0]])
            b1 = bias_variable([hidden_sizes[0]])

            # x_image = tf.reshape(x, [-1,28*28]) # maybe as sanity check
            h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

            # keep_prob = tf.placeholder(tf.float32)
            # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            W2 = weight_variable([hidden_sizes[0], hidden_sizes[1]])
            b2 = bias_variable([hidden_sizes[1]])
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

            W3 = weight_variable([hidden_sizes[1], 10])
            b3 = bias_variable([10])
            y_out = tf.matmul(h2, W3) + b3
            y_out_soft = softmax_T(y_out, T, tensor=True)         # temperature softmax
            y_out_hard = softmax_T(y_out, 1, tensor=True)         # temperature=1 softmax

            cross_entropy_soft = tf.reduce_mean(-tf.reduce_sum(y_soft_ * tf.log(y_out_soft), reduction_indices=[1]))
            cross_entropy_hard = tf.reduce_mean(-tf.reduce_sum(y_hard_ * tf.log(y_out_hard), reduction_indices=[1]))
            cross_entropy = ((T**2)*alpha*cross_entropy_soft) + ((1-alpha)*cross_entropy_hard)
            train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_out_hard,1), tf.argmax(y_hard_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess.run(tf.initialize_all_variables())

            iters_per_epoch = mnist.train.num_examples/batch_size
            for i in range(num_epochs*iters_per_epoch):
                batch = mnist.train.next_batch(batch_size,shuffle=False)
                if i%1000 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_hard_: batch[1]})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: batch[0],
                                          y_soft_: y_soft[(i%1100)*batch_size:((i%1100)+1)*batch_size],
                                          y_hard_: batch[1]})

            acc = accuracy_in_batches(mnist.test, accuracy, x, y_hard_, batch_size=batch_size)
            results_np_w_repeat[T_index, alpha_index, repeat] = acc
            avg_acc += acc
            sess.close()

        avg_acc /= num_repeat
        print("T: %f, alpha: %f, test accuracy %g"%(T, alpha, avg_acc))
        results_np[T_index, alpha_index] = avg_acc

np.savetxt('results_np.csv', results_np, delimiter=",")
np.savetxt('results_np_w_repeat.csv', results_np_w_repeat.reshape([Ts.shape[0], alphas.shape[0]*num_repeat]), delimiter=",")
