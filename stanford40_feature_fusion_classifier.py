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

def function_rearrange_labels(train_labels, NUMBER_OF_CLASSES, ground_truth_labels):
	cnt = 0
	for lbl in ground_truth_labels:
		train_labels[train_labels == lbl] = cnt
		cnt = cnt + 1
	return train_labels

GROUND_TRUTH_LABELS = np.array([1, 2,3, 4,5, 6, 7, 8, 9, 10])#np.array([1, 2, 4, 6])
PERCENTAGE_DROP_OUT1 = 0.2
NUMBER_OF_CLASSES = 10;
dim_feature = 500;
DIM_INPUT = NUMBER_OF_CLASSES * dim_feature#20 for sample data
DIM_FC1 = int(0.5 * DIM_INPUT)
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
	path_cross_features = '../data-stanford40/data1/'
	filename = path_cross_features + 'stanford40_50_500_cc1_cross_feat_ALL_CLASS_feat_fusion_clsfr__tr_10'
	tmp = scipy.io.loadmat(filename)
	data_train = tmp['cross_feautures_tr']
	
	filename = path_cross_features + 'stanford40_50_500_cc1_cross_feat_ALL_CLASS_feat_fusion_clsfr__ts_10'
	tmp = scipy.io.loadmat(filename)
	data_test = tmp['cross_feautures_ts']

	path_cross_features_labels = '../data-stanford40/data1/'
	filename = path_cross_features_labels + 'stanford40_50_500_feat_fusion_clsfr_train_labels'
	tmp = scipy.io.loadmat(filename)
	train_labels = tmp['train_labels']
	#TODO:rearrange labels
	train_labels = function_rearrange_labels(train_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS) 
	train_labels = train_labels - 1
	
	filename = path_cross_features_labels + 'stanford40_50_500_feat_fusion_clsfr_test_labels'
	tmp = scipy.io.loadmat(filename)
	test_labels = tmp['test_labels']
	#TODO:rearrange labels
	test_labels = function_rearrange_labels(test_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS)
	test_labels = test_labels - 1

all_classfet = [];
cnt = 0
#Rearrange cross features to make it classwise group
for cls in range(NUMBER_OF_CLASSES):
   st_aec = (cls) * dim_feature;
   end_aec = st_aec + dim_feature - 1;
   aef_train = data_train[:, st_aec:end_aec+1];
   aef_test = data_test[:, st_aec:end_aec+1];

   offset_cec = dim_feature * NUMBER_OF_CLASSES + (cls)* dim_feature * (NUMBER_OF_CLASSES - 1);
   st_cec = offset_cec;
   end_cec = st_cec + dim_feature * (NUMBER_OF_CLASSES - 1) - 1;
   cef_train = data_train[:, st_cec:end_cec+1];
   cef_test = data_test[:, st_cec:end_cec+1];

   classfet_train = np.hstack((aef_train, cef_train));
   classfet_test = np.hstack((aef_test, cef_test));

   if cnt == 0:
		all_classfet_train = classfet_train
		all_classfet_test = classfet_test
		cnt = 1
   else:
        all_classfet_train = np.hstack((all_classfet_train, classfet_train));
        all_classfet_test = np.hstack((all_classfet_test, classfet_test));
print all_classfet
#pdb.set_trace()

if SAMPLE_DATA:
		# Generate dummy data
		tr_data_1 = np.random.random((1000, 20))
		tr_data_2 = np.random.random((1000, 20))
		tr_data_3 = np.random.random((1000, 20))
		tr_data_4 = np.random.random((1000, 20))
		tr_labels = keras.utils.to_categorical(np.random.randint(NUMBER_OF_CLASSES, size=(1000, 1)), num_classes=NUMBER_OF_CLASSES)

		ts_data_1 = np.random.random((100, 20))
		ts_data_2 = np.random.random((100, 20))
		ts_data_3 = np.random.random((100, 20))
		ts_data_4 = np.random.random((100, 20))
		ts_labels = keras.utils.to_categorical(np.random.randint(NUMBER_OF_CLASSES, size=(100, 1)), num_classes=NUMBER_OF_CLASSES)
#pdb.set_trace()
else:
		end = 0
		tr_list = []
		ts_list = []
		for k in range(NUMBER_OF_CLASSES):
			start = end
			end = start + NUMBER_OF_CLASSES*dim_feature
			tr_data = all_classfet_train[:, start:end]
			ts_data = all_classfet_test[:, start:end] 
			tr_list.append(tr_data.tolist())
			ts_list.append(ts_data.tolist())
	
		tr_labels = keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_CLASSES)
		ts_labels = keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_CLASSES)

filename = path_cross_features + 'train_labels_categorial'		
scipy.io.savemat(filename, dict(train_labels = tr_labels))

filename = path_cross_features + 'test_labels_categorial'		
scipy.io.savemat(filename, dict(test_labels = ts_labels))

accuracy_array = []
#pdb.set_trace()
for itr in range(5):
	#Input layers
	input_1 = Input(shape=(DIM_INPUT,))
	input_2 = Input(shape=(DIM_INPUT,))
	input_3 = Input(shape=(DIM_INPUT,))
	input_4 = Input(shape=(DIM_INPUT,))
	input_5 = Input(shape=(DIM_INPUT,))
	input_6 = Input(shape=(DIM_INPUT,))
	input_7 = Input(shape=(DIM_INPUT,))
	input_8 = Input(shape=(DIM_INPUT,))
	input_9 = Input(shape=(DIM_INPUT,))
	input_10 = Input(shape=(DIM_INPUT,))

	#FC1 layers for each class
	fc1_1 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_1)
	fc1_2 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_2)
	fc1_3 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_3)
	fc1_4 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_4)
	fc1_5 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_5)
	fc1_6 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_6)
	fc1_7 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_7)
	fc1_8 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_8)
	fc1_9 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_9)
	fc1_10 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_10)

	#FC2layer for all classes
	concat_features = [fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, fc1_6, fc1_7, fc1_8, fc1_9, fc1_10]
	fc2_input = keras.layers.concatenate(concat_features)
	fc2_output = Dense(DIM_FC2, kernel_initializer='normal', activation='tanh')(fc2_input)
	drop_out1 = Dropout(PERCENTAGE_DROP_OUT1)(fc2_output)

	#Softmax layer
	output = Dense(NUMBER_OF_CLASSES, kernel_initializer='normal', activation='softmax')(drop_out1)

	#Define feature fusion model
	input_array = [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10]
	model_feature_fusion = Model(inputs=input_array, outputs=output)
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.09, nesterov=True)
	model_feature_fusion.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	train_data_array = [np.array(tr_list[0]), np.array(tr_list[1]), np.array(tr_list[2]), \
			 np.array(tr_list[3]), np.array(tr_list[4]), np.array(tr_list[5]), np.array(tr_list[6]), np.array(tr_list[7]) \
			, np.array(tr_list[8]), np.array(tr_list[9])]	

	model_feature_fusion.fit(train_data_array, tr_labels, epochs=10)
	model_feature_fusion.fit(train_data_array, tr_labels, epochs=10)

	test_data_array = [np.array(ts_list[0]), np.array(ts_list[1]), np.array(ts_list[2]), \
			 np.array(ts_list[3]), np.array(ts_list[4]), np.array(ts_list[5]), np.array(ts_list[6]), np.array(ts_list[7]) \
			, np.array(ts_list[8]), np.array(ts_list[9])]	
	score = model_feature_fusion.evaluate(test_data_array, ts_labels, batch_size=4)
	print "**********"
	print("%s: %.2f%%" % (model_feature_fusion.metrics_names[1], score[1]*100))
	print("%s: %.2f" % (model_feature_fusion.metrics_names[0], score[0]))
	print('AccuracyFusion: ', score)
	accuracy_array = np.hstack((accuracy_array, score[1]*100))
print 'Training data (%d, %d)'%(tr_labels.shape[0], tr_data_1.shape[1])
print 'Testing data (%d, %d)'%(ts_labels.shape[0], ts_data_1.shape[1])
print 'Mean Accuracy %.2f'%np.mean(accuracy_array, axis=0)
print 'Std Accuracy %.2f'%np.std(accuracy_array, axis=0)

