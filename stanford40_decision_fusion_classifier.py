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
import random
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

class lda_classifeir:
	predicted_labels = []
	cls = []
	def function(self):
		print("This is class lda_classifer")

def function_lda_classifier(train_data, train_labels, test_data, test_labels):
	pdb.set_trace()
	obj_lda_classifier = lda_classifier
	lda_classifier = LinearDiscriminantAnalysis()
	lda_classifier.fit(train_data, train_labels)
	LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage='auto',
        			      solver='eigen', store_covariance=False, tol=0.0001)
	predicted_labels = lda_classifier.predict(test_data)
	
	return obj_lda_classifier

class bal_train_samples:

	train_samples = []
	train_labels = []
	def function(self):
                print("This is class bal_train_samples")


def function_get_balanced_samples_for_training(cls, classfet_train, in_train_labels):
	indices = np.flatnonzero(in_train_labels == cls)
	n_pos_samples = np.size(indices)
	n_neg_samples_per_class = int(n_pos_samples / (NUMBER_OF_CLASSES - 1))
	train_samples = classfet_train[indices, :]
	train_labels = np.full((1, n_pos_samples), cls, int)
	for label in range(NUMBER_OF_CLASSES):
		if (label+1) != cls:
			ind = np.flatnonzero(in_train_labels == (1+ label))
			rand_sample_ind = random.sample(ind, n_neg_samples_per_class)
			train_samples = np.vstack((train_samples, classfet_train[rand_sample_ind, :]))
			train_labels = np.hstack((train_labels, np.full((1, n_neg_samples_per_class), (label + 1), int)))#in_train_labels[0, rand_sample_ind]))
	np.reshape(train_labels, (1, np.size(train_labels)))
	obj_bal_train_samples = bal_train_samples()
	obj_bal_train_samples.train_samples = train_samples
	obj_bal_train_samples.train_labels = train_labels
	return obj_bal_train_samples 
	
def function_get_binary_labels(labels, cls):
	labels_bin = np.ones(np.size(labels), int)
	for i in range(np.size(labels_bin) - 1):
		if labels[0, i] == cls:	
			labels_bin[i] = 0
	return labels_bin
	
#This function is used if class labels are not continuous like 1 to 10 but arbitrary labels like 2, 4, 8 etc.
def function_rearrange_labels(train_labels, NUMBER_OF_CLASSES, ground_truth_labels):
	cnt = 0
	for lbl in ground_truth_labels:
		train_labels[train_labels == lbl] = cnt
		cnt = cnt + 1
	return train_labels

GROUND_TRUTH_LABELS = np.array([1, 2,3, 4])#np.array([1, 2, 4, 6])
PERCENTAGE_DROP_OUT1 = 0.5
NUMBER_OF_CLASSES = 20
dim_feature = 500
DIM_INPUT = NUMBER_OF_CLASSES * dim_feature#20 for sample data
DIM_FC1 = int(0.5 * DIM_INPUT)
#DIM_FC1 = int(0.5 * dim_feature)
DIM_FC2 = int(0.5 * DIM_FC1)
SAMPLE_DATA = 0

if SAMPLE_DATA:
	data1 = np.array([1,1,1,2,2,2,3,3,3,4,4,4,12,12,12,13,13,13,14,14,14,\
			21,21,21,23,23,23,24,24,24,31,31,31,32,32,32,34,34,34,\
			41,41,41,42,42,42,43,43,43])
	data2 = np.array([1,1,1,2,2,2,3,3,3,4,4,4,12,12,12,13,13,13,14,14,14,\
			21,21,21,23,23,23,24,24,24,31,31,31,32,32,32,34,34,34,\
			41,41,41,42,42,42,43,43,43])
	data = np.vstack((data1, data2))
else:
	path_cross_features = '../data-stanford40/data3/'
	filename = path_cross_features + 'stanford40_50_500_aec_features_all_classes_tr_20'
	tmp = scipy.io.loadmat(filename)
	data_aec_train = tmp['cross_feautures_tr']
	
	filename = path_cross_features + 'stanford40_50_500_aec_features_all_classes_ts_20'
	tmp = scipy.io.loadmat(filename)
	data_aec_test = tmp['cross_feautures_ts']
	
	path_cross_features_labels = path_cross_features
	
	filename = path_cross_features_labels + 'stanford40_50_500_feat_fusion_clsfr_train_labels'
	tmp = scipy.io.loadmat(filename)
	train_labels = tmp['train_labels']
	#TODO:rearrange labels
	#train_labels = function_rearrange_labels(train_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS) 
	filename = path_cross_features_labels + 'stanford40_50_500_feat_fusion_clsfr_test_labels'
	tmp = scipy.io.loadmat(filename)
	test_labels = tmp['test_labels']
	#TODO:rearrange labels
	#test_labels = function_rearrange_labels(test_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS)

all_classfet = [];
cnt = 0
accuracy_array = []

#Rearrange cross features to make it classwise group
for cls in range(NUMBER_OF_CLASSES):
	tr_list = []
	ts_list = []
	filename = path_cross_features + 'stanford40_50_500_cec_features_class_tr_' + str(cls + 1) 
	print "Loading %s"%filename
	tmp = scipy.io.loadmat(filename)
	data_cec_train = tmp['cross_feautures_tr']
	
	filename = path_cross_features + 'stanford40_50_500_cec_features_class_ts_' + str(cls + 1) 
	print "Loading %s"%filename
	tmp = scipy.io.loadmat(filename)
	data_cec_test = tmp['cross_feautures_ts']
	
	st_aec = (cls) * dim_feature
	end_aec = st_aec + dim_feature - 1
	aef_train = data_aec_train[:, st_aec:end_aec+1]
	aef_test = data_aec_test[:, st_aec:end_aec+1]
	cef_train = data_cec_train#[:, st_cec:end_cec+1];
	cef_test = data_cec_test#[:, st_cec:end_cec+1];
	classfet_train = np.hstack((aef_train, cef_train))
	classfet_test = np.hstack((aef_test, cef_test))

	obj_bal_train_samples = function_get_balanced_samples_for_training(cls + 1, classfet_train, train_labels)	
	
	#tr_list.append(classfet_train.tolist())
	#ts_list.append(classfet_test.tolist())
	tr_list.append((obj_bal_train_samples.train_samples).tolist())
	ts_list.append(classfet_test.tolist())
	train_labels_bin = function_get_binary_labels(obj_bal_train_samples.train_labels, cls + 1)
	test_labels_bin = function_get_binary_labels(test_labels, cls + 1)
	tr_labels = keras.utils.to_categorical(train_labels_bin, num_classes=2)
	ts_labels = keras.utils.to_categorical(test_labels_bin, num_classes=2)

	filename = path_cross_features + 'train_labels_binary'		
	scipy.io.savemat(filename, dict(train_labels_bin = train_labels_bin))

	filename = path_cross_features + 'test_labels_binary'		
	scipy.io.savemat(filename, dict(test_labels_bin = test_labels_bin))
	
	filename = path_cross_features + 'train_labels_categorial'		
	scipy.io.savemat(filename, dict(train_labels_cate = tr_labels))

	filename = path_cross_features + 'test_labels_categorial'		
	scipy.io.savemat(filename, dict(test_labels_cate = ts_labels))

	obj_lda_classifier = function_lda_classifier(np.array(tr_list[0]), tr_labels, \
                                                     np.array(ts_list[0]), ts_labels)

	if 0:	
		for itr in range(1):
			#Input layers
			input_1 = Input(shape=(DIM_INPUT,))

			#FC1 layers for each class
			fc1_1 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_1)

			#FC2layer for all classes
			fc2_input = fc1_1#keras.layers.concatenate(concat_features)
			fc2_output = Dense(DIM_FC2, kernel_initializer='normal', activation='tanh')(fc2_input)
			drop_out1 = Dropout(PERCENTAGE_DROP_OUT1)(fc2_output)

			#Softmax layer
			output = Dense(2, kernel_initializer='normal', activation='softmax')(drop_out1)

			#Define feature fusion model
			input_array = [input_1]
			#input_array = [input_array1, input_array2]
			model_feature_fusion = Model(inputs=input_array, outputs=output)
			sgd = SGD(lr=0.001, decay=1e-6, momentum=0.01, nesterov=True)
			model_feature_fusion.compile(optimizer=sgd,
			      loss='binary_crossentropy',
			      metrics=['accuracy'])

			train_data_array = [np.array(tr_list[0])]
			model_feature_fusion.fit(train_data_array, tr_labels, epochs=50)
			test_data_array = [np.array(ts_list[0])]
			score = model_feature_fusion.evaluate(test_data_array, ts_labels)
			predicted_classes = model_feature_fusion.predict(test_data_array)
			filename = path_cross_features + 'predicted_classes_' + str(cls + 1)		
			scipy.io.savemat(filename, dict(predicted_classes = predicted_classes))
			print predicted_classes
			print "**********"
			print("%s: %.2f%%" % (model_feature_fusion.metrics_names[1], score[1]*100))
			print("%s: %.2f" % (model_feature_fusion.metrics_names[0], score[0]))
			print('AccuracyFusion: ', score)
			accuracy_array = np.hstack((accuracy_array, score[1]*100))
		time.sleep(2)
print 'Training data (%d, %d)'%(tr_labels.shape[0], tr_data_1.shape[1])
print 'Testing data (%d, %d)'%(ts_labels.shape[0], ts_data_1.shape[1])
print 'Mean Accuracy %.2f'%np.mean(accuracy_array, axis=0)
print 'Std Accuracy %.2f'%np.std(accuracy_array, axis=0)

