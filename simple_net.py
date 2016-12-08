import tensorflow as tf
import numpy as np
import helper_functions as hf


# RUN PARAMETERS
lrs = np.logspace(-7,-2,11)
Ts = np.logspace(np.log10(1),np.log10(20),11)
alphas, step = np.linspace(0,1,6,retstep=True)
num_repeat = 1                      # could be 10?
batch_size = 50
num_epochs = 2                      # could be changed... 5? 10? 20? write_logits.py uses 20.
hidden_sizes = [800, 800]           # could be increased when ready...


# Read train_logits
logits_cumbersome = np.loadtxt('Data/logits_mnist_tuned3.csv', delimiter=', ')


# Read input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def simple_network(sess, y_soft, lr, T, alpha, batch_size, num_epochs):
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
    y_out_soft = hf.softmax_T(y_out, T, tensor=True)    # temperature softmax
    y_out_hard = hf.softmax_T(y_out, 1, tensor=True)         # temperature=1 softmax

    cross_entropy_soft = tf.reduce_mean(-tf.reduce_sum(y_soft_, 1e-12 * tf.log(tf.maximum(y_out_soft, 1e-12)), reduction_indices=[1]))
    cross_entropy_hard = tf.reduce_mean(-tf.reduce_sum(y_hard_, 1e-12 * tf.log(tf.maximum(y_out_hard, 1e-12)), reduction_indices=[1]))
    cross_entropy = ((T**2)*alpha*cross_entropy_soft) + ((1-alpha)*cross_entropy_hard)

    # train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(lr)
    grads_and_vars = train_step.compute_gradients(cross_entropy, [W1,b1,W2,b2,W3,b3])
    capped_grads_and_vars = [(tf.minimum(tf.maximum(gv[0], 1e-8), 1e8), gv[1]) for gv in grads_and_vars]
    train_step.apply_gradients(capped_grads_and_vars)

    correct_prediction = tf.equal(tf.argmax(y_out_hard,1), tf.argmax(y_hard_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    iters_per_epoch = mnist.train.num_examples/batch_size
    for i in range(num_epochs*iters_per_epoch):
        batch = mnist.train.next_batch(batch_size,shuffle=False)
        # if i%1000 == 0:
        #     train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_hard_: batch[1]})
        #     print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0],
                                  y_soft_: y_soft[(i%iters_per_epoch)*batch_size:((i%iters_per_epoch)+1)*batch_size],
                                  y_hard_: batch[1]})

    return x, y_hard_, accuracy


results_np = np.zeros([lrs.shape[0], Ts.shape[0], alphas.shape[0]])
# results_np_w_repeat = np.zeros([lrs.shape[0], Ts.shape[0], alphas.shape[0], num_repeat])

best_val_acc = None
best_lr = None
best_T = None
best_alpha = None

# Tensorflow stuff
lr_index = -1
for lr in lrs:
    lr_index += 1

    T_index = -1
    for T in Ts:
        T_index += 1

        y_soft = hf.softmax_T(logits_cumbersome, T)  # temperature softmax

        alpha_index = -1
        for alpha in alphas:
            alpha_index += 1

            avg_val_acc  = 0
            for repeat in range(num_repeat):
                tf.reset_default_graph()
                sess = tf.InteractiveSession()

                x, y_hard_, accuracy = simple_network(sess, y_soft, lr, T, alpha, batch_size, num_epochs)

                val_acc  = hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
                avg_val_acc  += val_acc

                # results_np_w_repeat[lr_index, T_index, alpha_index, repeat] = val_acc

                sess.close()

            avg_val_acc /= num_repeat
            print 'lr: %.3e    T: %.4f    alpha: %.2f    val_acc: %.4f' % (lr, T, alpha, avg_val_acc)
            results_np[lr_index, T_index, alpha_index] = avg_val_acc

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_lr = lr
                best_T = T
                best_alpha = alpha

print '\n\nbest_lr: %.3e    best_T: %.4f    best_alpha: %.2f    best_val_acc: %.4f' % (best_lr, best_T, best_alpha, best_val_acc)

np.savetxt('Results/results_np.csv', results_np.reshape([lrs.shape[0], Ts.shape[0]*alphas.shape[0]]), delimiter=', ')
# np.savetxt('Results/results_np_w_repeat.csv', results_np_w_repeat.reshape([lrs.shape[0], Ts.shape[0]*alphas.shape[0]*num_repeat]), delimiter=', ')


################################################################
# Retrain the model with optimal hyperparameters for longer
num_epochs = 10

y_soft = hf.softmax_T(logits_cumbersome, best_T)  # temperature softmax

avg_train_acc = 0
avg_val_acc   = 0
avg_test_acc  = 0

for repeat in range(num_repeat):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x, y_hard_, accuracy = simple_network(sess, y_soft, best_lr, best_T, best_alpha, batch_size, num_epochs)

    avg_train_acc += hf.accuracy_in_batches(mnist.train, accuracy, x, y_hard_, batch_size=batch_size)
    avg_val_acc   += hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
    avg_test_acc  += hf.accuracy_in_batches(mnist.test, accuracy, x, y_hard_, batch_size=batch_size)

    sess.close()

avg_train_acc /= num_repeat
avg_val_acc   /= num_repeat
avg_test_acc  /= num_repeat

print '\n\nACCURACIES WITH A MODEL TRAINED WITH SOFT TARGETS AND OPTIMAL HYPERPARAMETERS'
print 'On training set:   %.4f' % avg_train_acc
print 'On validation set: %.4f' % avg_val_acc
print 'On test set:       %.4f' % avg_test_acc


################################################################
# Retrain the model with without soft targets for comparison
num_epochs = 10

y_soft = hf.softmax_T(logits_cumbersome, best_T)  # temperature softmax

avg_train_acc = 0
avg_val_acc   = 0
avg_test_acc  = 0

for repeat in range(num_repeat):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x, y_hard_, accuracy = simple_network(sess, y_soft, best_lr, 1.0, 0.0, batch_size, num_epochs)

    avg_train_acc += hf.accuracy_in_batches(mnist.train, accuracy, x, y_hard_, batch_size=batch_size)
    avg_val_acc   += hf.accuracy_in_batches(mnist.validation, accuracy, x, y_hard_, batch_size=batch_size)
    avg_test_acc  += hf.accuracy_in_batches(mnist.test, accuracy, x, y_hard_, batch_size=batch_size)

    sess.close()

avg_train_acc /= num_repeat
avg_val_acc   /= num_repeat
avg_test_acc  /= num_repeat

print '\n\nACCURACIES WITH A MODEL TRAINED WITHOUT SOFT TARGETS (ALL ELSE REMAINING THE SAME)'
print 'On training set:   %.4f' % avg_train_acc
print 'On validation set: %.4f' % avg_val_acc
print 'On test set:       %.4f' % avg_test_acc
