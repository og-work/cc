
#.............................oxfordpets_feature_fusion_classifier.py...................................


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

PERCENTAGE_DROP_OUT1 = 0.3
NUMBER_OF_CLASSES = 10
dim_feature = 50
DIM_INPUT = NUMBER_OF_CLASSES * dim_feature#20 for sample data
DIM_FC1 = int(0.5 * DIM_INPUT)
#DIM_FC1 = int(0.5 * dim_feature)
DIM_FC2 = int(0.5 * DIM_FC1)
EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE = 0.005 
DECAY = 1e-6 
MOMENTUM =0.09
#path_cross_features = '../data-oxfordpets/data1/'
path_cross_features = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-oxfordpets/data1_hemachandra/'
set_size = 697
SETWISE_TRAIN = 0

set1_s = 0
set1_e = set1_s + set_size
set2_s = set1_e
set2_e = set2_s + set_size
set3_s = set2_e
set3_e = set3_s + set_size
set4_s = set3_e
set4_e = set4_s + set_size
set5_s = set4_e
set5_e = set5_s + set_size
set6_s = set5_e
set6_e = set6_s + set_size

filename = path_cross_features + 'oxfordpets_50_500_aec_features_all_classes_tr_' + str(NUMBER_OF_CLASSES)
tmp = scipy.io.loadmat(filename)
data_aec_train = tmp['cross_feautures_tr']
print 'NUmber of training samples %d'%data_aec_train.shape[0]

filename = path_cross_features + 'oxfordpets_50_500_aec_features_all_classes_vl_' + str(NUMBER_OF_CLASSES)
tmp = scipy.io.loadmat(filename)
data_aec_valid = tmp['cross_feautures_val']
print 'NUmber of valid samples %d'%data_aec_valid.shape[0]

filename = path_cross_features + 'oxfordpets_50_500_aec_features_all_classes_ts_' + str(NUMBER_OF_CLASSES)
tmp = scipy.io.loadmat(filename)
data_aec_test = tmp['cross_feautures_ts']
print 'NUmber of testing samples %d'%data_aec_test.shape[0]

data_aec_train_valid = np.vstack((data_aec_train, data_aec_valid))
#data_aec_train_valid = data_aec_train
path_cross_features_labels = path_cross_features

filename = path_cross_features_labels + 'oxfordpets_50_500_feat_fusion_clsfr_train_labels'
tmp = scipy.io.loadmat(filename)
train_labels_all = tmp['train_labels']
train_labels_all = train_labels_all - 1
filename = path_cross_features_labels + 'oxfordpets_50_500_feat_fusion_clsfr_valid_labels'
tmp = scipy.io.loadmat(filename)
valid_labels_all = tmp['valid_labels']
valid_labels_all = valid_labels_all - 1

train_valid_labels_all = np.hstack((train_labels_all, valid_labels_all))
#train_valid_labels_all = train_labels_all
indices_dummy = np.arange(data_aec_train.shape[0] + data_aec_valid.shape[0])
#indices_dummy = np.arange(data_aec_train.shape[0])
shuffled_dummy_ind = random.sample(indices_dummy, (data_aec_train.shape[0] + data_aec_valid.shape[0]))
#shuffled_dummy_ind = random.sample(indices_dummy, (data_aec_train.shape[0]))
shuffled_train_valid_labels_all = train_valid_labels_all[0, shuffled_dummy_ind]
train_valid_labels1 = shuffled_train_valid_labels_all[set1_s:set1_e]
train_valid_labels2 = shuffled_train_valid_labels_all[set2_s:set2_e]
train_valid_labels3 = shuffled_train_valid_labels_all[set3_s:set3_e]
train_valid_labels4 = shuffled_train_valid_labels_all[set4_s:set4_e]
train_valid_labels5 = shuffled_train_valid_labels_all[set5_s:set5_e]

filename = path_cross_features_labels + 'oxfordpets_50_500_feat_fusion_clsfr_test_labels'
tmp = scipy.io.loadmat(filename)
test_labels_all = tmp['test_labels']
test_labels_all = test_labels_all - 1

print "Number of test samples are %d %d"%test_labels_all.shape
test_labels1 = test_labels_all[:, set1_s:set1_e]
test_labels2 = test_labels_all[:, set2_s:set2_e]
test_labels3 = test_labels_all[:, set3_s:set3_e]
test_labels4 = test_labels_all[:, set4_s:set4_e]
test_labels5 = test_labels_all[:, set5_s:set5_e]
test_labels6 = test_labels_all[:, set6_s:]
#TODO:rearrange labels
#test_labels = function_rearrange_labels(test_labels, NUMBER_OF_CLASSES, GROUND_TRUTH_LABELS)
all_classfet = [];
cnt = 0

#Rearrange cross features to make it classwise group
for cls in range(NUMBER_OF_CLASSES):
	filename = path_cross_features + 'oxfordpets_50_500_cec_features_class_tr_' + str(cls + 1) 
	print "Loading %s"%filename
	tmp = scipy.io.loadmat(filename)
	data_cec_train = tmp['cross_feautures_tr']
	
	filename = path_cross_features + 'oxfordpets_50_500_cec_features_class_vl_' + str(cls + 1) 
	print "Loading %s"%filename
	tmp = scipy.io.loadmat(filename)
	data_cec_valid = tmp['cross_feautures_val']
	
	data_cec_train_valid = np.vstack((data_cec_train, data_cec_valid))
	#data_cec_train_valid = data_cec_train
	filename = path_cross_features + 'oxfordpets_50_500_cec_features_class_ts_' + str(cls + 1) 
	print "Loading %s"%filename
	tmp = scipy.io.loadmat(filename)
	data_cec_test = tmp['cross_feautures_ts']
	
	st_aec = (cls) * dim_feature
	end_aec = st_aec + dim_feature - 1
	
	
	aef_train1 = data_aec_train_valid[shuffled_dummy_ind[set1_s:set1_e], st_aec:end_aec+1]
	#aef_valid_all = data_aec_valid[:, st_aec:end_aec+1]
	aef_test_all = data_aec_test[:, st_aec:end_aec+1]

	if SETWISE_TRAIN:
		aef_train2 = data_aec_train_valid[shuffled_dummy_ind[set2_s:set2_e], st_aec:end_aec+1]
		aef_train3 = data_aec_train_valid[shuffled_dummy_ind[set3_s:set3_e], st_aec:end_aec+1]
		aef_train4 = data_aec_train_valid[shuffled_dummy_ind[set4_s:set4_e], st_aec:end_aec+1]
		aef_train5 = data_aec_train_valid[shuffled_dummy_ind[set5_s:set5_e], st_aec:end_aec+1]
	
	aef_test1 = data_aec_test[set1_s:set1_e, st_aec:end_aec+1]
	if SETWISE_TRAIN:
		aef_test2 = data_aec_test[set2_s:set2_e, st_aec:end_aec+1]
		aef_test3 = data_aec_test[set3_s:set3_e, st_aec:end_aec+1]
		aef_test4 = data_aec_test[set4_s:set4_e, st_aec:end_aec+1]
		aef_test5 = data_aec_test[set5_s:set5_e, st_aec:end_aec+1]
		aef_test6 = data_aec_test[set6_s:, st_aec:end_aec+1]
	
	cef_train1 = data_cec_train_valid[shuffled_dummy_ind[set1_s:set1_e], :]
	#cef_valid_all = data_cec_valid
	cef_test_all = data_cec_test

	if SETWISE_TRAIN:
		cef_train2 = data_cec_train_valid[shuffled_dummy_ind[set2_s:set2_e], :]
		cef_train3 = data_cec_train_valid[shuffled_dummy_ind[set3_s:set3_e], :]
		cef_train4 = data_cec_train_valid[shuffled_dummy_ind[set4_s:set4_e], :]
		cef_train5 = data_cec_train_valid[shuffled_dummy_ind[set5_s:set5_e], :]
	
	cef_test1 = data_cec_test[set1_s:set1_e, :]
	if SETWISE_TRAIN:
		cef_test2 = data_cec_test[set2_s:set2_e:, :]
		cef_test3 = data_cec_test[set3_s:set3_e:, :]
		cef_test4 = data_cec_test[set4_s:set4_e:, :]
		cef_test5 = data_cec_test[set5_s:set5_e:, :]
		cef_test6 = data_cec_test[set6_s:, :]
	
	classfet_train1 = np.hstack((aef_train1, cef_train1))
	#classfet_valid_all = np.hstack((aef_valid_all, cef_valid_all))
	classfet_test_all = np.hstack((aef_test_all, cef_test_all))

	if SETWISE_TRAIN:
		classfet_train2 = np.hstack((aef_train2, cef_train2))
		classfet_train3 = np.hstack((aef_train3, cef_train3))
		classfet_train4 = np.hstack((aef_train4, cef_train4))
		classfet_train5 = np.hstack((aef_train5, cef_train5))

	classfet_test1 = np.hstack((aef_test1, cef_test1))
	if SETWISE_TRAIN:	
		classfet_test2 = np.hstack((aef_test2, cef_test2))
		classfet_test3 = np.hstack((aef_test3, cef_test3))
		classfet_test4 = np.hstack((aef_test4, cef_test4))
		classfet_test5 = np.hstack((aef_test5, cef_test5))
		classfet_test6 = np.hstack((aef_test6, cef_test6))
	#pdb.set_trace()	
	
	if cnt == 0:
		cnt = 1
		all_classfet_train1 = classfet_train1
		#all_classfet_valid_all = classfet_valid_all
		all_classfet_test_all = classfet_test_all
		if SETWISE_TRAIN:	
			all_classfet_train2 = classfet_train2
			all_classfet_train3 = classfet_train3
			all_classfet_train4 = classfet_train4
			all_classfet_train5 = classfet_train5
		all_classfet_test1 = classfet_test1
		if SETWISE_TRAIN:
			all_classfet_test2 = classfet_test2
			all_classfet_test3 = classfet_test3
			all_classfet_test4 = classfet_test4
			all_classfet_test5 = classfet_test5
			all_classfet_test6 = classfet_test6
	else:
		all_classfet_train1 = np.hstack((all_classfet_train1, classfet_train1));
		#all_classfet_valid_all = np.hstack((all_classfet_valid_all, classfet_valid_all));
		all_classfet_test_all = np.hstack((all_classfet_test_all, classfet_test_all));
		if SETWISE_TRAIN:
			all_classfet_train2 = np.hstack((all_classfet_train2, classfet_train2));
			all_classfet_train3 = np.hstack((all_classfet_train3, classfet_train3));
			all_classfet_train4 = np.hstack((all_classfet_train4, classfet_train4));
			all_classfet_train5 = np.hstack((all_classfet_train5, classfet_train5));

		all_classfet_test1 = np.hstack((all_classfet_test1, classfet_test1));
		if SETWISE_TRAIN:
			all_classfet_test2 = np.hstack((all_classfet_test2, classfet_test2));
			all_classfet_test3 = np.hstack((all_classfet_test3, classfet_test3));
			all_classfet_test4 = np.hstack((all_classfet_test4, classfet_test4));
			all_classfet_test5 = np.hstack((all_classfet_test5, classfet_test5));
			all_classfet_test6 = np.hstack((all_classfet_test6, classfet_test6));
	print "Stacking features for class %d"%(cls + 1)

#Input layers
print "Defining input layers"
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
if 0:
	input_11 = Input(shape=(DIM_INPUT,))
	input_12 = Input(shape=(DIM_INPUT,))
	input_13 = Input(shape=(DIM_INPUT,))
	input_14 = Input(shape=(DIM_INPUT,))
	input_15 = Input(shape=(DIM_INPUT,))
	input_16 = Input(shape=(DIM_INPUT,))
	input_17 = Input(shape=(DIM_INPUT,))
	input_18 = Input(shape=(DIM_INPUT,))
	input_19 = Input(shape=(DIM_INPUT,))
	input_20 = Input(shape=(DIM_INPUT,))
	input_21 = Input(shape=(DIM_INPUT,))
	input_22 = Input(shape=(DIM_INPUT,))
	input_23 = Input(shape=(DIM_INPUT,))
	input_24 = Input(shape=(DIM_INPUT,))
	input_25 = Input(shape=(DIM_INPUT,))
	input_26 = Input(shape=(DIM_INPUT,))
	input_27 = Input(shape=(DIM_INPUT,))
	input_28 = Input(shape=(DIM_INPUT,))
	input_29 = Input(shape=(DIM_INPUT,))
	input_30 = Input(shape=(DIM_INPUT,))
	input_31 = Input(shape=(DIM_INPUT,))
	input_32 = Input(shape=(DIM_INPUT,))
	input_33 = Input(shape=(DIM_INPUT,))
	input_34 = Input(shape=(DIM_INPUT,))
	input_35 = Input(shape=(DIM_INPUT,))
	input_36 = Input(shape=(DIM_INPUT,))
	input_37 = Input(shape=(DIM_INPUT,))

#FC1 layers for each class
print "Defining FC1 layers"
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
if 0:
	fc1_11 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_11)
	fc1_12 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_12)
	fc1_13 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_13)
	fc1_14 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_14)
	fc1_15 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_15)
	fc1_16 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_16)
	fc1_17 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_17)
	fc1_18 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_18)
	fc1_19 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_19)
	fc1_20 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_20)
	fc1_21 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_21)
	fc1_22 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_22)
	fc1_23 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_23)
	fc1_24 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_24)
	fc1_25 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_25)
	fc1_26 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_26)
	fc1_27 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_27)
	fc1_28 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_28)
	fc1_29 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_29)
	fc1_30 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_30)
	fc1_31 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_31)
	fc1_32 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_32)
	fc1_33 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_33)
	fc1_34 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_34)
	fc1_35 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_35)
	fc1_36 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_36)
	fc1_37 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_37)
#FC2layer for all classes
print "Defining FC2 layers"
concat_features = [fc1_1, fc1_2, fc1_3, fc1_4, fc1_5,\
		   fc1_6, fc1_7, fc1_8, fc1_9, fc1_10,\
		   #fc1_11, fc1_12, fc1_13, fc1_14, fc1_15,\
		   #fc1_16, fc1_17, fc1_18, fc1_19, fc1_20,\
		   #fc1_21, fc1_22, fc1_23, fc1_24, fc1_25,\
		   #fc1_26, fc1_27, fc1_28, fc1_29, fc1_30,\
		   #fc1_31, fc1_32, fc1_33, fc1_34, fc1_35,\
		   #fc1_36, fc1_37 \
		  ]
fc2_input = keras.layers.concatenate(concat_features)
fc2_output = Dense(DIM_FC2, kernel_initializer='normal', activation='tanh')(fc2_input)
drop_out1 = Dropout(PERCENTAGE_DROP_OUT1)(fc2_output)

#Softmax layer
print "Defining output layers"
output = Dense(NUMBER_OF_CLASSES, kernel_initializer='normal', activation='softmax')(drop_out1)

#Define feature fusion model
print "Defining input"
input_array = [input_1, input_2, input_3, input_4, input_5,\
	       input_6, input_7, input_8, input_9, input_10,\
	       #input_11, input_12, input_13, input_14, input_15,\
	       #input_16, input_17, input_18, input_19, input_20,\
	       #input_21, input_22, input_23, input_24, input_25,\
	       #input_26, input_27, input_28, input_29, input_30,\
	       #input_31, input_32, input_33, input_34, input_35,\
	       #input_36, input_37 \
	      ]

model_feature_fusion = Model(inputs=input_array, outputs=output)
sgd = SGD(lr=LEARNING_RATE, decay=DECAY, momentum = MOMENTUM, nesterov=True)

model_feature_fusion.compile(optimizer=sgd,
      loss='categorical_crossentropy',
      metrics=['accuracy'])


#Training
for itr in range(1):
	end = 0	
	tr_list = []
	if itr == 0:
		print "*************************"
		print "Training  set %d"%(itr+1)
		all_classfet_train_data = all_classfet_train1
		tr_labels = keras.utils.to_categorical(train_valid_labels1, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 1:
		print "*************************"
		print "Training  set %d"%(itr+1)
		all_classfet_train_data = all_classfet_train2
		tr_labels = keras.utils.to_categorical(train_valid_labels2, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 2:
		print "*************************"
		print "Training  set %d"%(itr+1)
		all_classfet_train_data = all_classfet_train3
		tr_labels = keras.utils.to_categorical(train_valid_labels3, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 3:
		print "*************************"
		print "Training  set %d"%(itr+1)
		all_classfet_train_data = all_classfet_train4
		tr_labels = keras.utils.to_categorical(train_valid_labels4, num_classes=NUMBER_OF_CLASSES)
	
	elif SETWISE_TRAIN == 1 and itr == 4:
		print "*************************"
		print "Training  set %d"%(itr+1)
		all_classfet_train_data = all_classfet_train5
		tr_labels = keras.utils.to_categorical(train_valid_labels5, num_classes=NUMBER_OF_CLASSES)
	
	for k in range(NUMBER_OF_CLASSES):
		start = end
		end = start + NUMBER_OF_CLASSES*dim_feature
		tr_data = all_classfet_train_data[:, start:end]
		tr_list.append(tr_data.tolist())

	print "Building train input"
	train_data_array = [np.array(tr_list[0]), np.array(tr_list[1]),\
			    np.array(tr_list[2]), np.array(tr_list[3]),\
			    np.array(tr_list[4]), np.array(tr_list[5]),\
			    np.array(tr_list[6]), np.array(tr_list[7]), \
			    np.array(tr_list[8]), np.array(tr_list[9]),\
			    #np.array(tr_list[10]), np.array(tr_list[11]),\
			    #np.array(tr_list[12]), np.array(tr_list[13]),\
			    #np.array(tr_list[14]), np.array(tr_list[15]), \
			    #np.array(tr_list[16]), np.array(tr_list[17]),\
			    #np.array(tr_list[18]), np.array(tr_list[19]),\
			    #np.array(tr_list[20]), np.array(tr_list[21]),\
			    #np.array(tr_list[22]), np.array(tr_list[23]),\
			    #np.array(tr_list[24]), np.array(tr_list[25]), \
			    #np.array(tr_list[26]), np.array(tr_list[27]),\
			    #np.array(tr_list[28]), np.array(tr_list[29]),\
			    #np.array(tr_list[30]), np.array(tr_list[31]),\
			    #np.array(tr_list[32]), np.array(tr_list[33]),\
			    #np.array(tr_list[34]), np.array(tr_list[35]), \
			    #np.array(tr_list[36]), \
			   ] 
	model_feature_fusion.fit(train_data_array, tr_labels, epochs=EPOCHS, shuffle=True, batch_size= BATCH_SIZE)
	#print 'Training data (%d, %d)'%(tr_labels.shape[0], tr_data.shape[1])
	
for itr in range(1):
	end = 0	
	tr_list = []
	ts_list = []
	if itr == 0:
		print "*************************"
		#print "validation  set %d"%(itr+1)
		#all_classfet_test_data = all_classfet_test_all
		#ts_labels = keras.utils.to_categorical(test_labels_all, num_classes=NUMBER_OF_CLASSES)
		all_classfet_test_data = all_classfet_test1
		ts_labels = keras.utils.to_categorical(test_labels1, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 1:
		print "*************************"
		print "Testing  set %d"%(itr+1)
		all_classfet_test_data = all_classfet_test2
		ts_labels = keras.utils.to_categorical(test_labels2, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 2:
		print "*************************"
		print "Testing  set %d"%(itr+1)
		all_classfet_test_data = all_classfet_test3
		ts_labels = keras.utils.to_categorical(test_labels3, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 3:
		print "*************************"
		print "Testing  set %d"%(itr+1)
		all_classfet_test_data = all_classfet_test4
		ts_labels = keras.utils.to_categorical(test_labels4, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 4:
		print "*************************"
		print "Testing  set %d"%(itr+1)
		all_classfet_test_data = all_classfet_test5
		ts_labels = keras.utils.to_categorical(test_labels5, num_classes=NUMBER_OF_CLASSES)
	elif SETWISE_TRAIN == 1 and itr == 5:
		print "*************************"
		print "Testing  set %d"%(itr+1)
		all_classfet_test_data = all_classfet_test6
		ts_labels = keras.utils.to_categorical(test_labels6, num_classes=NUMBER_OF_CLASSES)
	for k in range(NUMBER_OF_CLASSES):
		start = end
		end = start + NUMBER_OF_CLASSES*dim_feature
		ts_data = all_classfet_test_data[:, start:end] 
		ts_list.append(ts_data.tolist())


	print "Building test input"
	test_data_array = [np.array(ts_list[0]),  np.array(ts_list[1]),\
			    np.array(ts_list[2]), np.array(ts_list[3]),\
			    np.array(ts_list[4]), np.array(ts_list[5]),\
			    np.array(ts_list[6]), np.array(ts_list[7]),\
			    np.array(ts_list[8]), np.array(ts_list[9]),\
			    
			    #np.array(ts_list[10]), np.array(ts_list[11]),\
			    #np.array(ts_list[12]), np.array(ts_list[13]),\
			    #np.array(ts_list[14]), np.array(ts_list[15]),\
			    #np.array(ts_list[16]), np.array(ts_list[17]),\
			    #np.array(ts_list[18]), np.array(ts_list[19]),\
			    #np.array(ts_list[20]), np.array(ts_list[21]),\
			    #np.array(ts_list[22]), np.array(ts_list[23]),\
			    #np.array(ts_list[24]), np.array(ts_list[25]),\
			    #np.array(ts_list[26]), np.array(ts_list[27]),\
			    #np.array(ts_list[28]), np.array(ts_list[29]),\
			    #np.array(ts_list[30]), np.array(ts_list[31]),\
			    #np.array(ts_list[32]), np.array(ts_list[33]),\
			    #np.array(ts_list[34]), np.array(ts_list[35]),\
			    #np.array(ts_list[36]) \
			   ]

	score = model_feature_fusion.evaluate(test_data_array, ts_labels,  batch_size=1)

	print "**********"
	print("%s: %.2f%%" % (model_feature_fusion.metrics_names[1], score[1]*100))
	print("%s: %.2f" % (model_feature_fusion.metrics_names[0], score[0]))
	print('AccuracyFusion: ', score)
	print 'Testing data (%d, %d)'%(ts_labels.shape[0], ts_data.shape[1])

#print 'Mean Accuracy %.2f'%np.mean(accuracy_array, axis=0)
#print 'Std Accuracy %.2f'%np.std(accuracy_array, axis=0)

