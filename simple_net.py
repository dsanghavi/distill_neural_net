import tensorflow as tf
import numpy as np
import pickle

def softmax_T(logits, T):
	# logits dim = (N, C), T = float
	l1 = np.exp(logits/T)
	return l1/np.sum(l1,axis=1)[:,np.newaxis]


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Read logits
logits_cumbersome = np.loadtxt('logits_mnist1.txt')

# temperature softmax


# train small neural net ----------------------------------------------------------

#Read input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#interpolate?
# -----------------------------
Ts = np.logspace(0,np.log10(30),6)
alphas = np.linspace(0,1,11)


results = []
# Tensorflow stuff
for T in Ts:
	y_soft = softmax_T(logits_cumbersome,T)
	for alpha in alphas:
		y_new = (alpha*y_soft) + ((1-alpha)* mnist.train.labels)
		# assert np.mean(np.argmax(y_soft,axis=1) == np.argmax(mnist.train.labels,axis=1)) > 0.99
		for repeat in range(10):
			sess = tf.InteractiveSession()

			x = tf.placeholder(tf.float32, shape=[None, 784])
			y_ = tf.placeholder(tf.float32, shape=[None, 10])
			#Layers
			hidden_sizes = [800,800]
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


			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, y_))
			train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			sess.run(tf.initialize_all_variables())

			batch_size = 50
			num_epochs = 5
			for i in range(num_epochs*mnist.train.num_examples/batch_size):
				batch = mnist.train.next_batch(batch_size,shuffle=False)
				if i%1000 == 0:
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
					print("step %d, training accuracy %g"%(i, train_accuracy))
				train_step.run(feed_dict={x: batch[0], y_: y_new[(i%1100)*batch_size:((i%1100)+1)*batch_size]})

			acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
			print("alpha: %f, test accuracy %g"%(alpha, acc))
			results.append((T,alpha,acc))
f = open('results.pickle','wb')
pickle.dump(results,f)
f.close()