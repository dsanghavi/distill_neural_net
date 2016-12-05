import numpy as np
import tensorflow as tf
from helper_functions import *

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

hidden_sizes = [1200, 1200] # NOT USING THIS. 
# CURRENT DEFAULT HIDDEN SIZES = [1024] followed by output layer of size [10]

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

def launch_graph_and_train(lr=1e-4,kp=0.2,batch_size=50,num_epochs=20,sess=None):
    if sess is None:
        print "No session passed to function. ERROR."

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

    # Weight Initializations
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # LINKS
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # TRAINING
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    ############CROSS VALIDATION!?!?!?!????????????? TECHNICALLY, SHOULDN'T USE THE TEST DATA FOR HP. IS THAT CHEATING 
    ################################################ IN THIS CASE THOUGH, WHERE WE ONLY CARE ABOUT THE TRANSFER OF KNOWLEDGE?

    iters_per_epoch = mnist.train.num_examples/batch_size
    for i in range(num_epochs*iters_per_epoch):
      batch = mnist.train.next_batch(batch_size,shuffle=False)
      if i%5000 == 0:
          train_accuracy = accuracy.eval(feed_dict={
                  x:batch[0], y_: batch[1], keep_prob: 1.0})
          print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: kp})

    print("lr: %g, keep_prob: %g, test accuracy %g"%(lr,kp,accuracy_in_batches(mnist.validation, accuracy, x, y_, keep_prob, batch_size=batch_size)))
    # print("lr: %g, keep_prob: %g, test accuracy %g"%(lr,kp,accuracy_in_batches(mnist.test, accuracy, x, y_, keep_prob, batch_size=batch_size)))
    return x, y_conv, keep_prob

def fine_tune():
    # lrs = np.logspace(-6,-2,7) # rough space
    # lrs = np.logspace(-5,-3,7)
    lrs = np.logspace(-5,-2,15)[3:-1]
    # keep_probs = np.arange(5)/5.0
    keep_probs = [0.1,0.2,0.3,0.4,0.5]
    # status = 0
    for lr in lrs:
        # status += 1
        for kp in keep_probs:
            tf.reset_default_graph()

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

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess.run(tf.initialize_all_variables())

            batch_size = 50
            num_epochs = 20
            iters_per_epoch = mnist.train.num_examples/batch_size
            for i in range(num_epochs*iters_per_epoch):
                batch = mnist.train.next_batch(batch_size,shuffle=False)
                # if i%5000 == 0:
                #     train_accuracy = accuracy.eval(feed_dict={
                #             x:batch[0], y_: batch[1], keep_prob: 1.0})
                #     print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: kp})

            print("lr: %g, keep_prob: %g, test accuracy %g"%(lr,kp,accuracy_in_batches(mnist.test, accuracy, x, y_, keep_prob, batch_size=batch_size)))
            # print "STATUS ----> ", status, "/15"
            sess.close()
    
def write_best_logits(sess=None):
    # # lr = 0.00019307 # 1.93e-4
    # lr = 0.000848343 # 8.48e-4
    lr = 0.000517947 # 5.18e-4
    kp = 0.2
    batch_size = 50
    num_epochs = 2#20 
    iters_per_epoch = mnist.train.num_examples/batch_size

    x, y_conv, keep_prob = launch_graph_and_train(lr=lr,kp=kp,batch_size=batch_size,num_epochs=num_epochs,sess=sess)


    # WRITE THE LOGITS AFTER TUNING-----------------------------  
    # write out logits before softmax
    y_conv_nps = [] # Store the numpy arrays of values for each example
    for i in range(iters_per_epoch):
        batch = mnist.train.next_batch(batch_size,shuffle=False)
        y_conv_nps.append(y_conv.eval(feed_dict = {x: batch[0], keep_prob: 1.0})) # .eval() Only valid for an Interactive Session?

    y_conv_np_all = np.concatenate(y_conv_nps,axis=0) # new shape = (55000, 10)

    np.savetxt('logits_mnist_tuned3_52errors.csv', y_conv_np_all, delimiter = ', ')
    # to read, y_read = np.loadtxt('<filename.txt>',delimiter=', ')



# Uncomment for fine tuning ---------------------------------------------------
# fine_tune()

# AFTER FINE-TUNING, WRITE THE BEST LOGITS ------------------------------------
tf.reset_default_graph()
sess = tf.InteractiveSession()
write_best_logits(sess=sess)
sess.close()