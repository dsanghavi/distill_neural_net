import numpy as np
import tensorflow as tf
import helper_functions as hf

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

hidden_sizes = [1200, 1200]  # NOT USING THIS.
use_ensemble = True
num_models_in_ensemble = 2


# CURRENT DEFAULT HIDDEN SIZES = [1024] followed by output layer of size [10]

#######################
# TODO: WRITE LOGITS FOR TOP N NETWORKS. CAN EASILY USE ENSEMBLES AS ANOTHER EXPERIMENT.
# TODO: PRINT TIME TAKEN TO DECODE.
# TODO: MAYBE IF TIME PERMITS-PREPROCESS DATA FOR EACH CLASS AS ONE VS ALL FOR SPECIALIST NETS.
# TO REPORT: Distillation is a very strong form of regularizer
#######################

#### SHIFTED TO helper_functions.py
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)

#### SHIFTED TO helper_functions.py
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

#### SHIFTED TO helper_functions.py
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#### SHIFTED TO helper_functions.py
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')

#### SHIFTED TO helper_functions.py
# def variable_summaries(var, name):
#     """Attach a lot of summaries to a Tensor."""
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.scalar_summary('mean/' + name, mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.scalar_summary('stddev/' + name, stddev)
#         tf.scalar_summary('max/' + name, tf.reduce_max(var))
#         tf.scalar_summary('min/' + name, tf.reduce_min(var))
#         tf.histogram_summary(name, var)


#### SHIFTED TO helper_functions.py
# def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, operation=tf.matmul):
#     """Reusable code for making a simple neural net layer.
#     It does a matrix multiply, bias add, and then uses relu to nonlinearize.
#     It also sets up name scoping so that the resultant graph is easy to read,
#     and adds a number of summary ops.
#     """
#     # Adding a name scope ensures logical grouping of the layers in the graph.
#     with tf.name_scope(layer_name):
#         # Initialization
#         with tf.name_scope('weights'):
#             if operation is conv2d:
#                 weights = weight_variable(input_dim)
#             else:
#                 weights = weight_variable([input_dim, output_dim])
#             variable_summaries(weights, layer_name + '/weights')
#         with tf.name_scope('biases'):
#             biases = bias_variable([output_dim])
#             variable_summaries(biases, layer_name + '/biases')
#         # Operation
#         with tf.name_scope('Wx_plus_b'):
#             preactivate = operation(input_tensor, weights) + biases
#             tf.histogram_summary(layer_name + '/pre_activations', preactivate)
#         # Non linearity
#         activations = act(preactivate, name='activation')
#         tf.histogram_summary(layer_name + '/activations', activations)

#         return activations

def launch_graph_and_train_with_summaries(lr=1e-4, kp=0.2, batch_size=50, num_epochs=20, sess=None):
    if sess is None:
        print "No session passed to function. ERROR."

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x_input')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_input')

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.image_summary('input', x_image, 10)

    # Conv1 + Maxpool Layer
    h_conv1 = hf.nn_layer(x_image, [5, 5, 1, 32], 32, 'conv1', act=tf.nn.relu, operation=hf.conv2d)
    h_pool1 = hf.max_pool_2x2(h_conv1)

    # Conv2 + Maxpool Layer
    h_conv2 = hf.nn_layer(h_pool1, [5, 5, 32, 64], 64, 'conv2', act=tf.nn.relu, operation=hf.conv2d)
    h_pool2 = hf.max_pool_2x2(h_conv2)

    # Flatten
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # FC1 Layer
    h_fc1 = hf.nn_layer(h_pool2_flat, 7 * 7 * 64, 1024, 'fc1', act=tf.nn.relu, operation=tf.matmul)

    # Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2 Layer
    y_conv = hf.nn_layer(h_fc1_drop, 1024, 10, 'fc2', act=tf.identity, operation=tf.matmul)

    # TRAINING
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('tensorboard_logs/mnist_logs' + '/train',
                                          sess.graph)
    # Test writer actually logs validation accuracies
    val_writer = tf.train.SummaryWriter('tensorboard_logs/mnist_logs' + '/val')

    sess.run(tf.initialize_all_variables())

    iters_per_epoch = mnist.train.num_examples / batch_size
    for i in range(num_epochs * iters_per_epoch):
        batch = mnist.train.next_batch(batch_size, shuffle=False)
        # Log test stats
        if i % 50 == 0:
            summary, val_accuracy = sess.run([merged, accuracy],
                                             feed_dict={x: mnist.validation.images, y_: mnist.validation.labels,
                                                        keep_prob: 1.0})
            val_writer.add_summary(summary, i)
            print("step %d, intermediate val accuracy %g" % (i, val_accuracy))

        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: kp})
        train_writer.add_summary(summary, i)

    train_writer.close()
    val_writer.close()

    val_acc = hf.accuracy_in_batches(mnist.validation, accuracy, x, y_, keep_prob, batch_size=batch_size)
    print("lr: %g, keep_prob: %g, val_accuracy: %g" % (lr, kp, val_acc))

    return x, y_, y_conv, keep_prob, accuracy, val_acc


def launch_graph_and_train(lr=1e-4, kp=0.2, batch_size=50, num_epochs=20, sess=None):
    if sess is None:
        print "No session passed to function. ERROR."

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Weight Initializations
    W_conv1 = hf.weight_variable([5, 5, 1, 32])
    b_conv1 = hf.bias_variable([32])

    W_conv2 = hf.weight_variable([5, 5, 32, 64])
    b_conv2 = hf.bias_variable([64])

    W_fc1 = hf.weight_variable([7 * 7 * 64, 1024])
    b_fc1 = hf.bias_variable([1024])

    W_fc2 = hf.weight_variable([1024, 10])
    b_fc2 = hf.bias_variable([10])

    # LINKS
    h_conv1 = tf.nn.relu(hf.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = hf.max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(hf.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = hf.max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # TRAINING
    softmax = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)  # should be dim 10
    cross_entropy = tf.reduce_mean(softmax)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    iters_per_epoch = mnist.train.num_examples / batch_size
    for i in range(num_epochs * iters_per_epoch):
        batch = mnist.train.next_batch(batch_size, shuffle=False)
        # if i%50 == 0:
        #     train_accuracy = accuracy.eval(feed_dict={
        #             x:batch[0], y_: batch[1], keep_prob: 1.0})
        #     print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: kp})

    val_acc = hf.accuracy_in_batches(mnist.validation, accuracy, x, y_, keep_prob, batch_size=batch_size)
    print("lr: %g, keep_prob: %g, val_accuracy %g" % (lr, kp, val_acc))

    return x, y_, y_conv, keep_prob, accuracy, val_acc


def fine_tune():
    best_val_acc = None
    best_lr = None
    best_kp = None

    batch_size = 50
    num_epochs = 2
    iters_per_epoch = mnist.train.num_examples / batch_size

    # lrs = np.logspace(-6,-2,7) # rough space
    # lrs = np.logspace(-5,-3,7)
    # lrs = np.logspace(-5, -2, 15)[3:-1]
    lrs = np.logspace(-4, -3, 2)
    # keep_probs = np.arange(5)/5.0
    # keep_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    keep_probs = [0.5]

    results = {}
    for lr in lrs:
        for kp in keep_probs:
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            _, _, _, _, _, val_acc = launch_graph_and_train(lr=lr, kp=kp, batch_size=batch_size, num_epochs=num_epochs,
                                                            sess=sess)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_kp = kp

            results[(lr, kp)] = val_acc

            sess.close()

    return best_lr, best_kp, results


def write_best_logits(sess=None, lr=0.000517947, kp=0.2, file_postfix='ensemble0'):
    # # lr = 0.00019307 # 1.93e-4
    # lr = 0.000848343 # 8.48e-4
    # lr = 0.000517947 # 5.18e-4
    # kp = 0.2
    batch_size = 50
    num_epochs = 20
    iters_per_epoch = mnist.train.num_examples / batch_size

    x, y_, y_conv, keep_prob, accuracy_tf, _ = launch_graph_and_train_with_summaries(lr=lr, kp=kp,
                                                                                     batch_size=batch_size,
                                                                                     num_epochs=num_epochs, sess=sess)

    # WRITE THE LOGITS AFTER TUNING-----------------------------
    # write out logits before softmax
    y_conv_nps = []  # Store the numpy arrays of values for each example
    for i in range(iters_per_epoch):
        batch = mnist.train.next_batch(batch_size, shuffle=False)
        y_conv_nps.append(
            y_conv.eval(feed_dict={x: batch[0], keep_prob: 1.0}))  # .eval() Only valid for an Interactive Session?

    y_conv_np_all = np.concatenate(y_conv_nps, axis=0)  # new shape = (55000, 10)

    np.savetxt('logits_mnist_tuned_' + file_postfix + '.csv', y_conv_np_all, delimiter=', ')
    # to read, y_read = np.loadtxt('<filename.txt>',delimiter=', ')

    test_accuracy = hf.accuracy_in_batches(mnist.test, accuracy_tf, x, y_, batch_size=batch_size)
    print '\nTEST ACCURACY ' + file_postfix + ' \n', test_accuracy

    if use_ensemble:
        # Store test logits for overall evaluation
        y_conv_test_nps = []  # Store numpy arrays, like above
        for i in range(mnist.test.num_examples / batch_size):
            batch = mnist.train.next_batch(batch_size, shuffle=False)
            y_conv_test_nps.append(
                y_conv.eval(feed_dict={x: batch[0], keep_prob: 1.0}))  # .eval() Only valid for an Interactive Session?
        y_conv_test_np_all = np.concatenate(y_conv_test_nps, axis=0)  # new shape = (10000, 10)
        votes = (y_conv_test_np_all.T > np.max(y_conv_test_np_all, axis=1)).T

        return votes


# Uncomment for fine tuning ---------------------------------------------------
best_lr, best_kp, results = fine_tune()

# AFTER FINE-TUNING, WRITE THE BEST LOGITS ------------------------------------
if use_ensemble:
    votes_total = np.zeros([mnist.test.num_examples, 10])
    sorted_results = sorted(results, key=lambda x: -results[x])
    for i in range(num_models_in_ensemble):
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        votes = write_best_logits(sess=sess, lr=sorted_results[0][0], kp=sorted_results[0][1],
                                  file_postfix='ensemble' + str(i))  # ADD FILENAME POSTFIX VARIABLE
        votes_total += votes  # votes.shape
        sess.close()

    predictions = (votes_total.T > np.max(votes_total, axis=1)).T
    # ACCURACY CALCULATION
    correct = np.sum(predictions * mnist.test.labels)  # Each of dim (10000,10)
    print "\nFINAL TEST ACCURACY FOR ENSEMBLE = ", correct / mnist.test.num_examples

else:
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    write_best_logits(sess=sess, lr=best_lr, kp=best_kp, file_postfix='')
    sess.close()
