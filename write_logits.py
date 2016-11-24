import numpy as np
import tensorflow as tf
from helper_functions import *

hidden_sizes = [1200, 1200]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# keep_prob = tf.placeholder(tf.float32)
#
# #Layers
# W1 = weight_variable([28 * 28, hidden_sizes[0]])
# b1 = bias_variable([hidden_sizes[0]])
#
# # x_image = tf.reshape(x, [-1,28*28]) # maybe as sanity check
# h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
#
# h1_dropped = tf.nn.dropout(h1, keep_prob)
#
# W2 = weight_variable([hidden_sizes[0], hidden_sizes[1]])
# b2 = bias_variable([hidden_sizes[1]])
# h2 = tf.nn.relu(tf.matmul(h1_dropped, W2) + b2)
#
# h2_dropped = tf.nn.dropout(h2, keep_prob)
#
# W3 = weight_variable([hidden_sizes[1], 10])
# b3 = bias_variable([10])
# y_conv = tf.matmul(h2_dropped, W3) + b3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

batch_size = 50
num_epochs = 20
iters_per_epoch = mnist.train.num_examples/batch_size
for i in range(num_epochs*iters_per_epoch):
    batch = mnist.train.next_batch(batch_size,shuffle=False)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy_in_batches(mnist.test, accuracy, x, y_, keep_prob, batch_size=batch_size))

# write out logits before softmax
y_conv_nps = []
for i in range(iters_per_epoch):
    batch = mnist.train.next_batch(batch_size,shuffle=False)
    y_conv_nps.append(y_conv.eval(feed_dict = {x: batch[0], keep_prob: 1.0}))

y_conv_np = np.concatenate(y_conv_nps,axis=0) # new shape = (55000, 10)

np.savetxt('logits_mnist1.txt', y_conv_np)
# np.savetxt('logits_mnist2.txt', y_conv_np)
# to read, y_read = np.loadtxt('<filename.txt>')
