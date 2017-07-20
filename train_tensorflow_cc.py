import numpy as np
import pdb
import scipy.io
import tensorflow as tf
import math
import time
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets

class train_tf_cc_input:
	cc1_input_train_perm = np.array([])
	cc1_output_train_perm = np.array([])
	cc1_input_valid_perm = np.array([])
	cc1_output_valid_perm = np.array([])
	cc1_input_train = np.array([])
	cc1_input_valid = np.array([])
	cc1_input_test = np.array([])
	dimension_hidden_layer1 = []
	EPOCHS_CC = []

    	def function(self):
        	print("This is train_tensorflow_cc_input class")

class train_tf_cc_output:
	decoded_data_train_cc1 = np.array([])
	encoded_data_train_cc1 = np.array([])
	decoded_data_valid_cc1 = np.array([])
	encoded_data_valid_cc1 = np.array([])
	decoded_data_test_cc1 = np.array([])
	encoded_data_test_cc1 = np.array([])

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
		batch_xs = obj_train_tf_cc_input.cc1_input_train_perm[sample][:]
		batch_ys = obj_train_tf_cc_input.cc1_output_train_perm[sample][:]
		sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
		if i % 50 == 0:
			print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})
	cc1_end = time.time()
	cc1_time = cc1_end - cc1_start

  #Get cc features for training  and validation samples
	obj_train_tf_cc_output = train_tf_cc_output()
	obj_train_tf_cc_output.decoded_data_train_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.cc1_input_train})
	obj_train_tf_cc_output.encoded_data_train_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.cc1_input_train})

	obj_train_tf_cc_output.decoded_data_valid_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.cc1_input_valid})
	obj_train_tf_cc_output.encoded_data_valid_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.cc1_input_valid})
	
  #Get cc features for testing samples
	obj_train_tf_cc_output.encoded_data_test_cc1 = sess.run(h, feed_dict={x: obj_train_tf_cc_input.cc1_input_test})
	obj_train_tf_cc_output.decoded_data_test_cc1 = sess.run(y, feed_dict={x: obj_train_tf_cc_input.cc1_input_test})
	
	return obj_train_tf_cc_output


class classifier_input:

	train_data = []
	test_data = []
	train_labels = []
	test_labels = []
	number_of_train_classes = []
	number_of_samples_per_class_test = []
	number_of_samples_per_class_train = []
	
	def function(self):
		print("This is a classifier_input object")


class classifier_output:

	accuracy = []
	predicted_labels = []
	
	def function(self):
		print("This is a classifier_output object")

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    RANDOM_SEED = 42
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def function_train_classifier_for_cc(obj_classifier_input):

	obj_classifier_input.train_labels = obj_classifier_input.train_labels - 1
	labels_train = keras.utils.to_categorical(obj_classifier_input.train_labels, num_classes = obj_classifier_input.number_of_train_classes)
	obj_classifier_input.test_labels = obj_classifier_input.test_labels - 1
	labels_test = keras.utils.to_categorical(obj_classifier_input.test_labels, num_classes = obj_classifier_input.number_of_train_classes)
	print("Train data and labels")
	print(obj_classifier_input.train_data.shape, labels_train.shape)
	print("Test data and labels")
	print(obj_classifier_input.test_data.shape, labels_test.shape)
	
	pdb.set_trace()
	RANDOM_SEED = 42
	tf.set_random_seed(RANDOM_SEED)
	train_X = obj_classifier_input.train_data
	test_X = obj_classifier_input.test_data
	train_y = labels_train
	test_y = labels_test
	pdb.set_trace()
	
	
	# Layer's sizes
	x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
	h_size = 256                # Number of hidden nodes
	y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

	# Symbols
	X = tf.placeholder("float", shape=[None, x_size])
	y = tf.placeholder("float", shape=[None, y_size])

	# Weight initializations
	w_1 = init_weights((x_size, h_size))
	w_2 = init_weights((h_size, y_size))

	# Forward propagation
	yhat    = forwardprop(X, w_1, w_2)
	predict = tf.argmax(yhat, axis=1)

	# Backward propagation
	cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
	updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

	# Run SGD
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	for epoch in range(100):
	# Train with each example
		for i in range(len(train_X)):
		    sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

	train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
				 sess.run(predict, feed_dict={X: train_X, y: train_y}))
	test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
				 sess.run(predict, feed_dict={X: test_X, y: test_y}))

	print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
	      % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
	pdb.set_trace()
	sess.close()

if 0:
	# define baseline model
	def baseline_model(dim_input, dim_hidden_layer, dim_output):
		# create model
		model = Sequential()
		model.add(Dense(dim_hidden_layer, input_dim=dim_input, init='normal', activation='relu'))
		model.add(Dense(dim_output, init='normal', activation='sigmoid'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


	def function_train_classifier_for_cc(obj_classifier_input):
		seed = 7
		np.random.seed(seed)
		
	obj_classifier_input.train_labels = obj_classifier_input.train_labels - 1
	labels_train = keras.utils.to_categorical(obj_classifier_input.train_labels, num_classes = obj_classifier_input.number_of_train_classes)
	labels_test = keras.utils.to_categorical(obj_classifier_input.test_labels, num_classes = obj_classifier_input.number_of_train_classes)

	NUMBER_OF_EPOCHS = 100
	BATCH_SIZE = 128
	DIM_HIDDEN_LAYER_1 = int(0.7 * obj_classifier_input.train_data.shape[1])

	estimator = KerasClassifier(build_fn=baseline_model(obj_classifier_input.train_data.shape[1], \
		 DIM_HIDDEN_LAYER_1, obj_classifier_input.number_of_train_classes), nb_epoch=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, verbose=0)	
	kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
	results = cross_val_score(estimator,  obj_classifier_input.train_data, labels_train, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	#Find accuracy
	obj_classifier_output = classifier_output()
	obj_classifier_output.predicted_labels = predicted_labels

	return obj_classifier_output
