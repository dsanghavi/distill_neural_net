import tensorflow as tf
import numpy as np
import helper_functions as hf
import sys
import os
import time
import datetime


################################################################
# RUN PARAMETERS
lr = 1e-5                  # best_lr, comes from simple_net
T = 1.8206                      # best_T, comes from simple_net
alpha = 0.80                # best_alpha, comes from simple_net
num_repeat = 5
batch_size = 50
num_epochs = 10
hidden_sizes = [800, 800]


################################################################
# !!! SHOULD REVERSE THIS AT THE END OF THIS SCRIPT !!!
# Set up printing out a log (redirects all prints to the file)
orig_stdout = sys.stdout
f_log_name = 'Results/' + os.path.basename(__file__) \
                        + '_' \
                        + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S') \
                        + '.log'
logger = hf.Logger(f_log_name)
sys.stdout = logger


################################################################
# LOG THE RUN PARAMETERS
print '\nRUN PARAMETERS'
print 'lr =', lr
print 'T =', T
print 'alpha =', alpha
print 'num_repeat =', num_repeat
print 'batch_size =', batch_size
print 'num_epochs =', num_epochs
print 'hidden_sizes =', hidden_sizes
print '\n'


################################################################
# LOAD DATA

# Load data
train_features    = np.loadtxt('Data/mnist_w_7_8_train_features.csv', delimiter=', ')
train_hard_labels = np.loadtxt('Data/mnist_w_7_8_train_labels.csv', dtype='int', delimiter=', ')
train_soft_labels = np.loadtxt('Data/mnist_w_7_8_train_logits.csv', delimiter=', ')

# USE MNIST.VALIDATION AND MNIST.TEST AS USUAL
# BUT USE THE FOLLOWING TRUNCATED SET FOR TRAINING
truncated_size = 20000
train_features    = train_features[:truncated_size]
train_hard_labels = train_hard_labels[:truncated_size]
train_soft_labels = train_soft_labels[:truncated_size]

################################################################
# NEURAL NETWORK
print '\nTraining the model...'

iters_per_epoch = np.ceil(float(train_features.shape[0]) / batch_size).astype(int)

y_soft = hf.softmax_T(train_soft_labels, T)

tf.reset_default_graph()
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_soft_ = tf.placeholder(tf.float32, shape=[None, 10])
y_hard_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = hf.weight_variable([28 * 28, hidden_sizes[0]])
b1 = hf.bias_variable([hidden_sizes[0]])
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = hf.weight_variable([hidden_sizes[0], hidden_sizes[1]])
b2 = hf.bias_variable([hidden_sizes[1]])
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = hf.weight_variable([hidden_sizes[1], 10])
b3 = hf.bias_variable([10])
y_out = tf.matmul(h2, W3) + b3
y_out_soft = hf.softmax_T(y_out, T, tensor=True)         # temperature softmax
y_out_hard = hf.softmax_T(y_out, 1, tensor=True)         # temperature=1 softmax

cross_entropy_soft = tf.reduce_mean(-tf.reduce_sum(y_soft_ * tf.log(tf.maximum(y_out_soft, 1e-12)), reduction_indices=[1]))
cross_entropy_hard = tf.reduce_mean(-tf.reduce_sum(y_hard_ * tf.log(tf.maximum(y_out_hard, 1e-12)), reduction_indices=[1]))
cross_entropy = ((T**2)*alpha*cross_entropy_soft) + ((1-alpha)*cross_entropy_hard)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, iters_per_epoch, 0.9, staircase=True)
train_step_opt = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = train_step_opt.compute_gradients(cross_entropy, [W1,b1,W2,b2,W3,b3])
capped_grads_and_vars = [(tf.sign(gv[0])*tf.minimum(tf.maximum(tf.abs(gv[0]), 1e-8), 1e8), gv[1]) for gv in grads_and_vars]
train_step = train_step_opt.apply_gradients(capped_grads_and_vars)

correct_prediction = tf.equal(tf.argmax(y_out_hard,1), tf.argmax(y_hard_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(num_epochs*iters_per_epoch):
    start_index = (i % iters_per_epoch) * batch_size
    end_index = ((i % iters_per_epoch) + 1) * batch_size
    if end_index > train_features.shape[0]:
        end_index = train_features.shape[0]

    features_batch = train_features[start_index:end_index]
    soft_labels_batch = y_soft[start_index:end_index]
    hard_labels_batch = train_hard_labels[start_index:end_index]

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:features_batch, y_hard_: hard_labels_batch})
        print("step %d, training accuracy %g"%(i, train_accuracy))
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


print '\n\nERRORS ON TEST SET'
test_features, test_labels = mnist.test.next_batch(10000, shuffle=False)
predictions = np.argmax(y_out_hard.eval(feed_dict={x: test_features, y_soft_: test_labels, y_hard_: test_labels}), 1)
truth = np.argmax(test_labels, 1)
confusion_matrix = np.zeros([test_labels.shape[1], test_labels.shape[1]], dtype='int')
for i in range(predictions.shape[0]):
    confusion_matrix[truth[i], predictions[i]] += 1
for i in range(test_labels.shape[1]):
    num_correctly_predicted = confusion_matrix[i, i]
    num_actually_were = np.sum(confusion_matrix[i, :])
    num_wrongly_predicted = num_actually_were - num_correctly_predicted
    print str(i) + ':\t', num_wrongly_predicted, '/', num_actually_were, '\t', confusion_matrix[i, :]

################################################################
# TODO: change bias for affected classes and test again
################################################################

sess.close()

################################################################
# !!! RESETTING STDOUT LOGGING !!!
# Stop redirecting pring out to log
logger.close_log()
sys.stdout = orig_stdout
