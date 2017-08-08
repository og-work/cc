#cnn_features_classifier.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
#import pandas

import scipy.io
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pdb


def get_session(gpu_fraction=0.3):
	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	
	KTF.set_session(get_session())


def function_rearrange_labels(train_labels, NUMBER_OF_CLASSES, ground_truth_labels):
	cnt = 0
	for lbl in ground_truth_labels:
		train_labels[train_labels == lbl] = cnt
		cnt = cnt + 1
	return train_labels

GROUND_TRUTH_LABELS = np.array([8,12,17,23,24])#np.array([1, 2, 4, 6])
DIM_INPUT = 4096#20 for sample data
PERCENTAGE_DROP_OUT1 = 0.2
DIM_FC1 = int(0.5 * DIM_INPUT)
DIM_FC2 = int(0.5 * DIM_FC1)
SAMPLE_DATA = 0
NUMBER_OF_CLASSES = 5;
dim_feature = 500;

if SAMPLE_DATA:
	data1 = np.array([1,1,1,2,2,2,3,3,3,4,4,4,12,12,12,13,13,13,14,14,14,\
			21,21,21,23,23,23,24,24,24,31,31,31,32,32,32,34,34,34,\
			41,41,41,42,42,42,43,43,43])
	data2 = np.array([1,1,1,2,2,2,3,3,3,4,4,4,12,12,12,13,13,13,14,14,14,\
			21,21,21,23,23,23,24,24,24,31,31,31,32,32,32,34,34,34,\
			41,41,41,42,42,42,43,43,43])
	data = np.vstack((data1, data2))
else:
	path_cross_features = 'data4/'
	filename = path_cross_features + 'apy50_500__cnn_svm_4096_dim_train_data'
	tmp = scipy.io.loadmat(filename)
	train_data = tmp['train_data']
	
	filename = path_cross_features + 'apy50_500__cnn_svm_4096_dim_test_data'
	tmp = scipy.io.loadmat(filename)
	test_data = tmp['test_data']

	filename = path_cross_features + 'apy50_500__feat_fusion_clsfr_train_labels'
	tmp = scipy.io.loadmat(filename)
	train_labels = tmp['train_labels']
	#TODO:rearrange labels
	train_labels = function_rearrange_labels(train_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS) 
	train_labels = train_labels - 1
	
	filename = path_cross_features + 'apy50_500__feat_fusion_clsfr_test_labels'
	tmp = scipy.io.loadmat(filename)
	test_labels = tmp['test_labels']
	#TODO:rearrange labels
	test_labels = function_rearrange_labels(test_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS)
	test_labels = test_labels - 1

	tr_labels = keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_CLASSES)
	ts_labels = keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_CLASSES)

filename = path_cross_features + 'train_labels_categorial'		
scipy.io.savemat(filename, dict(train_labels = tr_labels))

filename = path_cross_features + 'test_labels_categorial'		
scipy.io.savemat(filename, dict(test_labels = ts_labels))

#pdb.set_trace()
#Input layers
input_1 = Input(shape=(DIM_INPUT,))

#FC1 layers for each class
fc1 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_1)

#FC2 layer for all classes
fc2 = Dense(DIM_FC2, kernel_initializer='normal', activation='tanh')(fc1)
drop_out1 = Dropout(PERCENTAGE_DROP_OUT1)(fc2)

#Softmax layer
output = Dense(NUMBER_OF_CLASSES, kernel_initializer='normal', activation='softmax')(drop_out1)

#Define feature fusion model
model_feature_fusion = Model(inputs=input_1, outputs=output)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.09, nesterov=True)
model_feature_fusion.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_feature_fusion.fit(train_data, tr_labels, epochs=50)
score = model_feature_fusion.evaluate(test_data, ts_labels, batch_size=32)
print "**********"
print("%s: %.2f%%" % (model_feature_fusion.metrics_names[1], score[1]*100))
print("%s: %.2f" % (model_feature_fusion.metrics_names[0], score[0]))
print('Accuracy: ', score)
print 'Training data (%d, %d)'%(tr_labels.shape[0], tr_data_1.shape[1])
print 'Testing data (%d, %d)'%(ts_labels.shape[0], ts_data_1.shape[1])
