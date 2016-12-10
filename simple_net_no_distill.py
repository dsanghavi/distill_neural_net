import tensorflow as tf
import numpy as np
import helper_functions as hf
import sys
import os
import time
import datetime


################################################################
# RUN PARAMETERS
lrs = np.logspace(-6,-3,10)
num_repeat = 1                      # could be 10?
batch_size = 50
num_epochs = 5                      # could be changed... 5? 10? 20? write_logits.py uses 20.
hidden_sizes = [800, 800]           # could be increased when ready...


################################################################
# !!! SHOULD RESET THIS AT THE END OF THIS SCRIPT !!!
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
print 'lrs =', lrs
print 'num_repeat =', num_repeat
print 'batch_size =', batch_size
print 'num_epochs =', num_epochs
print 'hidden_sizes =', hidden_sizes
print '\n'


################################################################
# LOAD DATA

# Read input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# USE MNIST.VALIDATION AND MNIST.TEST AS USUAL
# BUT USE THE FOLLOWING TRUNCATED SET FOR TRAINING
truncated_size = 20000
mnist_train_truncated = mnist.train.next_batch(batch_size=truncated_size,shuffle=False)


################################################################
# NEURAL NETWORK

def simple_network_no_distill(sess, lr, batch_size, num_epochs):
    iters_per_epoch = mnist_train_truncated[0].shape[0] / batch_size

    x = tf.placeholder(tf.float32, shape=[None, 784])
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
    y_out_hard = hf.softmax_T(y_out, 1, tensor=True)         # temperature=1 softmax

    cross_entropy_hard = tf.reduce_mean(-tf.reduce_sum(y_hard_ * tf.log(tf.maximum(y_out_hard, 1e-12)), reduction_indices=[1]))
    cross_entropy = cross_entropy_hard

    # train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
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
        x_batch = mnist_train_truncated[0][(i % iters_per_epoch) * batch_size:((i % iters_per_epoch) + 1) * batch_size]
        y_hard_batch = mnist_train_truncated[1][(i % iters_per_epoch) * batch_size:((i % iters_per_epoch) + 1) * batch_size]
        # # TO PRINT WEIGHTS AND GRADIENTS
        # if i % 10 == 0:
        #     print 'W1\t\tmax: %.3e\t\tmin: %.3e\t\tmean: %.3e' % (tf.reduce_max(W1).eval(), tf.reduce_min(W1).eval(), tf.reduce_mean(W1).eval())
        #     print 'W2\t\tmax: %.3e\t\tmin: %.3e\t\tmean: %.3e' % (tf.reduce_max(W2).eval(), tf.reduce_min(W2).eval(), tf.reduce_mean(W2).eval())
        #     print 'W3\t\tmax: %.3e\t\tmin: %.3e\t\tmean: %.3e' % (tf.reduce_max(W3).eval(), tf.reduce_min(W3).eval(), tf.reduce_mean(W3).eval())
        #     for gv in sess.run(grads_and_vars, feed_dict={x: batch[0],
        #                           y_soft_: y_soft[(i%iters_per_epoch)*batch_size:((i%iters_per_epoch)+1)*batch_size],
        #                           y_hard_: batch[1]}):
        #         print tf.reduce_max(gv[0]).eval(), tf.reduce_min(gv[0]).eval(), tf.reduce_mean(gv[0]).eval()
        # # TO PRINT TRAINING AND VALIDATION ACCURACIES WHILE TRAINING
        # if i%1000 == 0:
        #     train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_hard_: batch[1]})
        #     val_acc = hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
        #     print("step %d\t\ttraining accuracy %.4f\t\tval acc %.4f"%(i, train_accuracy, val_acc))
        train_step.run(feed_dict={x: x_batch, y_hard_: y_hard_batch})

    return x, y_hard_, accuracy


# ################################################################
# # HYPERPARAMETER OPTIMIZATION
# print '\nHyperparameter optimization...\n'
#
# best_val_acc = None
# best_lr = None
#
# # Tensorflow stuff
# lr_index = -1
# for lr in lrs:
#     lr_index += 1
#
#     avg_val_acc  = 0
#     for repeat in range(num_repeat):
#         tf.reset_default_graph()
#         sess = tf.InteractiveSession()
#
#         x, y_hard_, accuracy = simple_network_no_distill(sess, lr, batch_size, num_epochs)
#
#         val_acc  = hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
#         avg_val_acc  += val_acc
#
#         sess.close()
#
#     avg_val_acc /= num_repeat
#     print 'lr: %.3e    val_acc: %.4f' % (lr, avg_val_acc)
#
#     if avg_val_acc > best_val_acc:
#         best_val_acc = avg_val_acc
#         best_lr = lr
#
# print '\n\nbest_lr: %.3e    best_val_acc: %.4f' % (best_lr, best_val_acc)


################################################################
# TRAIN A MODEL WITH OPTIMAL HYPERPARAMETERS FOR LONGER
best_lr = 1e-5
num_epochs = 5
print '\n\nTraining a model with optimal hyperparameters for', num_epochs, 'epochs...'

avg_train_acc = 0
avg_val_acc   = 0
avg_test_acc  = 0

for repeat in range(num_repeat):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x, y_hard_, accuracy = simple_network_no_distill(sess, best_lr, batch_size, num_epochs)

    avg_train_acc += hf.accuracy_in_batches_alt(mnist_train_truncated[0], mnist_train_truncated[1], accuracy, x, y_hard_, batch_size=batch_size)
    avg_val_acc   += hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
    avg_test_acc  += hf.accuracy_in_batches(mnist.test, accuracy, x, y_hard_, batch_size=batch_size)

    sess.close()

avg_train_acc /= num_repeat
avg_val_acc   /= num_repeat
avg_test_acc  /= num_repeat

print '\n\nACCURACIES WITH A TUNED MODEL TRAINED WITHOUT SOFT TARGETS'
print 'On training set:   %.4f' % avg_train_acc
print 'On validation set: %.4f' % avg_val_acc
print 'On test set:       %.4f' % avg_test_acc


################################################################
# !!! RESETTING STDOUT LOGGING !!!
# Stop redirecting pring out to log
logger.close_log()
sys.stdout = orig_stdout
