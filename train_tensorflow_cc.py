import numpy as np
import pdb
import scipy.io
import tensorflow as tf
import math
import time

class train_tf_cc_input:
	cc1_input_train = np.array([])
	cc1_output_train = np.array([])
	cc1_input_valid = np.array([])
	cc1_output_valid = np.array([])
	dimension_hidden_layer1 = []
	EPOCHS_CC = []

    	def function(self):
        	print("This is train_tensorflow_cc_input class")

class train_tf_cc_output:
	decoded_data_train_cc1 = np.array([])
	encoded_data_train_cc1 = np.array([])
	decoded_data_valid_cc1 = np.array([])
	encoded_data_valid_cc1 = np.array([])

    	def function(self):
        	print("This is train_tensorflow_cc_output class")

def function_train_tensorflow_cc(obj_train_tf_cc_input):

	n_samp, input_dim = (obj_train_tf_cc_input.cc1_input_train).shape
	n_hidden = obj_train_tf_cc_input.dimension_hidden_layer1
	x = tf.placeholder("float", [None, input_dim])
    # Weights and biases to hidden layer
	Wh = tf.Variable(tf.random_uniform((input_dim, n_hidden), -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
	bh = tf.Variable(tf.zeros([n_hidden]))
	h = tf.nn.relu(tf.matmul(x,Wh) + bh)
    # Weights and biases to hidden layer
    #Wo = tf.transpose(Wh) # tied weights
	Wo = tf.Variable(tf.random_uniform((n_hidden, input_dim), -1.0 / math.sqrt(n_hidden), 1.0 / math.sqrt(n_hidden)))
	bo = tf.Variable(tf.zeros([input_dim]))
	y = tf.nn.sigmoid(tf.matmul(h,Wo) + bo)
    # Objective functions
	y_ = tf.placeholder("float", [None,input_dim])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	meansq = tf.reduce_mean(tf.square(y_-y))
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)
	
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	n_rounds = obj_train_tf_cc_input.EPOCHS_CC
	batch_size = min(50, n_samp)
	#pdb.set_trace()
	cc1_start = time.time()
	for i in range(n_rounds):
		sample = np.random.randint(n_samp, size=batch_size)
		batch_xs = obj_train_tf_cc_input.cc1_input_train[sample][:]
		batch_ys = obj_train_tf_cc_input.cc1_output_train[sample][:]
		sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
		if i % 50 == 0:
			print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})
	cc1_end = time.time()
	cc1_time = cc1_end - cc1_start

    #Get cc features for training samples
	obj_train_tf_cc_output = train_tf_cc_output()
	obj_train_tf_cc_output.decoded_data_train_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.cc1_input_train})
	obj_train_tf_cc_output.encoded_data_train_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.cc1_input_train})
	obj_train_tf_cc_output.decoded_data_valid_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.cc1_input_valid})
	obj_train_tf_cc_output.encoded_data_valid_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.cc1_input_valid})
	
	return obj_train_tf_cc_output
