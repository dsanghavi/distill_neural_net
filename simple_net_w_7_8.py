import tensorflow as tf
import numpy as np
import helper_functions as hf


# RUN PARAMETERS
T = 4                      # best_T, comes from simple_net
alpha = 0.6                # best_alpha, comes from simple_net
num_repeat = 5             # could be 10
batch_size = 50            # no need to change I guess
num_epochs = 2             # could be changed... 5? 10? 20? write_logits.py uses 20.
hidden_sizes = [800, 800]  # could be increased when ready...


# Load data
train_features    = np.loadtxt('Data/mnist_w_7_8_train_features.csv', delimiter=', ')
train_hard_labels = np.loadtxt('Data/mnist_w_7_8_train_labels.csv', dtype='int', delimiter=', ')
train_soft_labels = np.loadtxt('Data/mnist_w_7_8_train_logits.csv', delimiter=', ')


# train small neural net ----------------------------------------------------------

tf.reset_default_graph()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_soft_ = tf.placeholder(tf.float32, shape=[None, 10])
y_hard_ = tf.placeholder(tf.float32, shape=[None, 10])

#Layers
W1 = hf.weight_variable([28 * 28, hidden_sizes[0]])
b1 = hf.bias_variable([hidden_sizes[0]])

# x_image = tf.reshape(x, [-1,28*28]) # maybe as sanity check
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W2 = hf.weight_variable([hidden_sizes[0], hidden_sizes[1]])
b2 = hf.bias_variable([hidden_sizes[1]])
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = hf.weight_variable([hidden_sizes[1], 10])
b3 = hf.bias_variable([10])
y_out = tf.matmul(h2, W3) + b3
y_out_soft = hf.softmax_T(y_out, T, tensor=True)         # temperature softmax
y_out_hard = hf.softmax_T(y_out, 1, tensor=True)         # temperature=1 softmax

cross_entropy_soft = tf.reduce_mean(-tf.reduce_sum(y_soft_ * tf.log(y_out_soft), reduction_indices=[1]))
cross_entropy_hard = tf.reduce_mean(-tf.reduce_sum(y_hard_ * tf.log(y_out_hard), reduction_indices=[1]))
cross_entropy = ((T**2)*alpha*cross_entropy_soft) + ((1-alpha)*cross_entropy_hard)
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out_hard,1), tf.argmax(y_hard_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

iters_per_epoch = np.ceil(float(train_features.shape[0]) / batch_size).astype(int)
for i in range(num_epochs*iters_per_epoch):
    start_index = (i % iters_per_epoch) * batch_size
    end_index = ((i % iters_per_epoch) + 1) * batch_size
    if end_index > train_features.shape[0]:
        end_index = train_features.shape[0]

    features_batch = train_features[start_index:end_index]
    soft_labels_batch = train_soft_labels[start_index:end_index]
    hard_labels_batch = train_hard_labels[start_index:end_index]

    # if i%1000 == 0:
    #     train_accuracy = accuracy.eval(feed_dict={x:features_batch, y_hard_: hard_labels_batch})
    #     print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: features_batch, y_soft_: soft_labels_batch, y_hard_: hard_labels_batch})

train_acc = hf.accuracy_in_batches_alt(train_features, train_hard_labels, accuracy, x, y_hard_, batch_size=batch_size)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
val_acc   = hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
test_acc  = hf.accuracy_in_batches(mnist.test, accuracy, x, y_hard_, batch_size=batch_size)

print '\n\nACCURACIES'
print 'On training set:   %.4f' % train_acc
print 'On validation set: %.4f' % val_acc
print 'On test set:       %.4f' % test_acc

################################################################
# TODO: change bias for affected classes and test again
################################################################

sess.close()
