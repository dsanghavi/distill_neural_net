import tensorflow as tf
import numpy as np
from helper_functions import *


# Load data
features    = np.loadtxt('Data/mnist_w_7_8_features.csv', delimiter=', ')
hard_labels = np.loadtxt('Data/mnist_w_7_8_labels.csv', dtype='int', delimiter=', ')
soft_labels = np.loadtxt('Data/mnist_w_7_8_logits.csv', delimiter=', ')


# train small neural net ----------------------------------------------------------

# RUN PARAMETERS
Ts = np.logspace(0,np.log10(30),11) # range should be fine, increase intervals when ready
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
    y_soft = softmax_T(soft_labels,T)     # temperature softmax
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
            sess.run(tf.global_variables_initializer())

            iters_per_epoch = np.ceil(float(features.shape[0])/batch_size).astype(int)
            for i in range(num_epochs*iters_per_epoch):
                start_index = (i % iters_per_epoch) * batch_size
                end_index = ((i % iters_per_epoch) + 1) * batch_size
                if end_index > features.shape[0]:
                    end_index = features.shape[0]

                features_batch = features[start_index:end_index]
                soft_labels_batch = soft_labels[start_index:end_index]
                hard_labels_batch = hard_labels[start_index:end_index]

                if i%1000 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:features_batch, y_hard_: hard_labels_batch})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: features_batch, y_soft_: soft_labels_batch, y_hard_: hard_labels_batch})

            acc = accuracy_in_batches_alt(features, hard_labels, accuracy, x, y_hard_, batch_size=batch_size)
            results_np_w_repeat[T_index, alpha_index, repeat] = acc
            avg_acc += acc
            sess.close()

        avg_acc /= num_repeat
        print("T: %f, alpha: %f, test accuracy %g"%(T, alpha, avg_acc))
        results_np[T_index, alpha_index] = avg_acc

np.savetxt('Results/results_np_w_7_8.csv', results_np, delimiter=', ')
np.savetxt('Results/results_np_w_repeat_w_7_8.csv', results_np_w_repeat.reshape([Ts.shape[0], alphas.shape[0]*num_repeat]), delimiter=', ')
