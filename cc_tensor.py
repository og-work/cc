#
#
#..........cc_tensor.py..........
#
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from get_data_for_cc import function_get_training_data_cc, input_cc, output_cc, function_normalise_data
from get_data_for_cc import input_data, function_get_input_data
from train_tensorflow_cc import train_tf_cc_input, train_tf_cc_output, function_train_tensorflow_cc
from train_tensorflow_cc import classifier_input, classifier_output, function_train_classifier_for_cc
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import scipy.io
import matplotlib
from keras.datasets import mnist
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import pdb
print "*****************************************************************************************************************************************"

EPOCHS = 1000
EPOCHS_CC = 5000
BATCH_SIZE = 128
BATCH_SIZE_CC = 128
TR_TS_VA_SPLIT = np.array([0.6, 0.2, 0.2])
MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES = 50
NOISE_FACTOR = 0.01
INCREASE_FACTOR = 1
dataset_list = ['sample', 'apy']
DATASET_INDEX = 0
system_list = ['desktop', 'laptop']
SYSTEM_INDEX = 0
DATA_SAVE_PATH = '/home/SharedData/omkar/'

#Prepare encoder model...................
if DATASET_INDEX == 0:
	dimension_hidden_layer1 = 3
	dimension_hidden_layer2 = 2
	dimension_hidden_layer3 = 1
	REDUCED_DIMENSION_VISUAL_FEATURE = 5
else:
	dimension_hidden_layer1 = 300
	dimension_hidden_layer2 = 30
	dimension_hidden_layer3 = 20
	REDUCED_DIMENSION_VISUAL_FEATURE = 1000

	
#Load input data..................................
obj_input_data = input_data()
obj_input_data.dataset_name = dataset_list[DATASET_INDEX]
obj_input_data.system_type = system_list[SYSTEM_INDEX]

obj_input_data = function_get_input_data(obj_input_data)
#pdb.set_trace()
visual_features_dataset_ori = obj_input_data.visual_features_dataset
train_class_labels = obj_input_data.train_class_labels
test_class_labels = obj_input_data.test_class_labels
attributes = obj_input_data.attributes
dataset_labels = obj_input_data.dataset_labels

#PCA
visual_features_dataset_norm = StandardScaler().fit_transform(visual_features_dataset_ori)
visual_features_dataset = PCA(n_components = REDUCED_DIMENSION_VISUAL_FEATURE).fit_transform(visual_features_dataset_norm)

number_of_train_classes = np.size(train_class_labels)
number_of_test_classes = np.size(test_class_labels)
dimension_visual_data = visual_features_dataset.shape[1]
number_of_samples_dataset = visual_features_dataset.shape[0]
dimension_attributes = attributes.shape[1]
number_of_classes = attributes.shape[0]
print "Dataset visual features shape is: %d X %d" % visual_features_dataset.shape
print "Dimension of visual data: %d" %dimension_visual_data
print "Number of dataset samples: %d" %number_of_samples_dataset
print "Dimension of attributes: %d" %dimension_attributes
print "Number of classes: %d" %number_of_classes
print "Train classes are"
print(train_class_labels)
print "Test classes are"
print(test_class_labels)

scipy.io.savemat('data/cnn_features.mat', \
                dict(cnn_features = visual_features_dataset))
print("Saved cnn features")


#......................cc.......................
number_of_cc = number_of_train_classes * number_of_train_classes - number_of_train_classes
cross_coders_train_data_input = []
cross_coders_train_data_output = []

#Get mean feature vector for each class
mean_feature_mat = np.empty((0, dimension_visual_data), float)
number_of_samples_per_class_train = []
number_of_samples_per_class_test = []
number_of_samples_per_class_valid = []

for classI in train_class_labels:
	ind = np.flatnonzero(dataset_labels == classI)
	classI_features = visual_features_dataset[ind.astype(int), :]
	mean_feature = classI_features.mean(0)
	mean_feature_mat = np.append(mean_feature_mat, mean_feature.reshape(1, dimension_visual_data), axis = 0)	
	number_of_samples_per_class_train.append(int(TR_TS_VA_SPLIT[0] * np.size(ind))) 
	number_of_samples_per_class_test.append(int(TR_TS_VA_SPLIT[2] * np.size(ind))) 
	number_of_samples_per_class_valid.append(int(TR_TS_VA_SPLIT[1] * np.size(ind))) 

file_name = DATA_SAVE_PATH + 'data/' + dataset_list[DATASET_INDEX] + '_mean_features_' + str(dimension_hidden_layer1) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE)
#scipy.io.savemat(file_name, dict(mean_visula_features = mean_feature_mat))

#Empty list for containing all cross features for all classes
encoded_cross_features_all_classes_train = []
encoded_cross_features_all_classes_labels_train = []
encoded_cross_features_all_classes_test = []
encoded_cross_features_all_classes_labels_test = []

cc_start = time.time() 

cnt1 = 0
for classI in train_class_labels:
	cross_features_classI_train = []
	cross_features_classI_labels_train = []
	cross_features_classI_test = []
	cross_features_classI_labels_test = []
	number_of_classI_samples_train = 0
	number_of_classI_samples_test = 0
	cnt = 0
	for classJ in train_class_labels:
		print "**************************************"
		if classI != classJ:
			
			#cc1..............................................
			
			#Get data for training cc.........................
			obj_input_cc = input_cc()
			obj_input_cc.classI = classI
			obj_input_cc.classJ = classJ
			obj_input_cc.visual_features = function_normalise_data(visual_features_dataset)
			obj_input_cc.train_valid_split = TR_TS_VA_SPLIT
			obj_input_cc.dataset_labels = dataset_labels
			
			obj_cc1_train_valid_data = function_get_training_data_cc(obj_input_cc)
			cc1_input_train = obj_cc1_train_valid_data.input_train_perm
			#cc1_input_train_ori = function_normalise_data(cc1_input_train)
			cc1_input_train = np.tile(cc1_input_train, (INCREASE_FACTOR, 1))
			cc1_input_train = cc1_input_train + NOISE_FACTOR * np.random.normal(0, 1, cc1_input_train.shape)
			cc1_input_train = function_normalise_data(cc1_input_train)

			cc1_output_train = obj_cc1_train_valid_data.output_train_perm
			#cc1_output_train_ori = function_normalise_data(cc1_output_train)
			cc1_output_train = np.tile(cc1_output_train, (INCREASE_FACTOR, 1))
			cc1_output_train = cc1_output_train + NOISE_FACTOR * np.random.normal(0, 1, cc1_output_train.shape)
			cc1_output_train = function_normalise_data(cc1_output_train)
			cc1_start = time.time()
		
			#Train tensorflow cc.....................................
			obj_train_tf_cc_input = train_tf_cc_input()
			obj_train_tf_cc_input.cc1_input_train_perm = cc1_input_train
			obj_train_tf_cc_input.cc1_output_train_perm = cc1_output_train
			obj_train_tf_cc_input.cc1_input_valid_perm = obj_cc1_train_valid_data.input_valid_perm
			obj_train_tf_cc_input.cc1_output_valid_perm = obj_cc1_train_valid_data.output_valid_perm
			obj_train_tf_cc_input.cc1_input_train = obj_cc1_train_valid_data.input_train
			obj_train_tf_cc_input.cc1_input_valid = obj_cc1_train_valid_data.input_valid
			obj_train_tf_cc_input.cc1_input_test = obj_cc1_train_valid_data.input_test
			obj_train_tf_cc_input.dimension_hidden_layer1 = dimension_hidden_layer1
			obj_train_tf_cc_input.EPOCHS_CC = EPOCHS_CC
			obj_train_tf_cc_output = function_train_tensorflow_cc(obj_train_tf_cc_input)
			
			#Save data for cc1..................................
			if 0:	
					file_name = DATA_SAVE_PATH + 'data/' + dataset_list[DATASET_INDEX] + '_' + str(dimension_hidden_layer1) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) \
									+ '_cc1_data_part1_' + str(classI) + '_' + str(classJ) + '.mat'		
					scipy.io.savemat(file_name, \
						dict(encoded_data_train_cc1 = obj_train_tf_cc_output.encoded_data_train_cc1,\
							 decoded_data_train_cc1 = obj_train_tf_cc_output.decoded_data_train_cc1))
					print "Saved data for cc1: %s" %file_name
			
					file_name = DATA_SAVE_PATH + 'data/' + dataset_list[DATASET_INDEX] + '_'+ str(dimension_hidden_layer1) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) + \
									'_cc1_data_part2_' + str(classI) + '_' + str(classJ) + '.mat'		
					scipy.io.savemat(file_name, \
						dict(cc1_input_train = cc1_input_train, \
							 cc1_output_train = cc1_output_train))
					print "Save data for cc1: %s" %file_name
					 
					file_name = DATA_SAVE_PATH + 'data/' + dataset_list[DATASET_INDEX] + '_' + str(dimension_hidden_layer1) + '_' + str(REDUCED_DIMENSION_VISUAL_FEATURE) + \
							'_cc1_data_part3_' + str(classI) + '_' + str(classJ) + '.mat'		
					scipy.io.savemat(file_name, \
						dict(cc1_input_train_ori = cc1_input_train_ori,\
							 cc1_output_train_ori = cc1_output_train_ori, \
									 indices_output_sample_train = obj_cc1_train_valid_data.indices_ouput_samples_train,\
					 indices_input_sample_train = obj_cc1_train_valid_data.indices_input_samples_train))
					print "Save data for cc1: %s" %file_name

					file_name = DATA_SAVE_PATH + 'data/' + dataset_list[DATASET_INDEX] + '_' + str(dimension_hidden_layer1) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) \
							+ '_cc1_data_part4_' + str(classI) + '_' + str(classJ) + '.mat'		
					scipy.io.savemat(file_name, \
					dict(encoded_data_valid_cc1 = obj_train_tf_cc_output.encoded_data_valid_cc1,\
				     decoded_data_valid_cc1 = obj_train_tf_cc_output.decoded_data_valid_cc1))
					print "Saved data for cc1: %s" %file_name
			
					file_name = DATA_SAVE_PATH + 'data/' + dataset_list[DATASET_INDEX] + '_'+ str(dimension_hidden_layer1) + '_'+ str(REDUCED_DIMENSION_VISUAL_FEATURE) + \
							'_cc1_data_part5_' + str(classI) + '_' + str(classJ) + '.mat'		
					scipy.io.savemat(file_name, \
						dict(cc1_input_valid = obj_cc1_train_valid_data.input_valid, \
					 	cc1_output_valid = obj_cc1_train_valid_data.output_valid))
					print "Save data for cc1: %s" %file_name
			
			#pdb.set_trace()			
			#Concat encoded cross features:
			#cross_features_classI_train.append(np.ndarray.tolist(obj_train_tf_cc_output.encoded_data_train_cc1))
			if cnt == 0:
				cross_features_classI_train = obj_train_tf_cc_output.encoded_data_train_cc1
				cross_features_classI_test = obj_train_tf_cc_output.encoded_data_test_cc1
				cnt  = 1
			else:
				cross_features_classI_train = np.hstack((cross_features_classI_train, obj_train_tf_cc_output.encoded_data_train_cc1))
				cross_features_classI_test = np.hstack((cross_features_classI_test, obj_train_tf_cc_output.encoded_data_test_cc1))
			print("here...........")
#			pdb.set_trace()

	number_of_classI_samples_train = obj_train_tf_cc_output.encoded_data_train_cc1.shape[0] 
	classI_labels_array_train = np.empty(number_of_classI_samples_train)
	classI_labels_array_train.fill(classI)
	
	number_of_classI_samples_test = obj_train_tf_cc_output.encoded_data_test_cc1.shape[0] 
	classI_labels_array_test = np.empty(number_of_classI_samples_test)
	classI_labels_array_test.fill(classI)
	
	if cnt1 == 0:
		cross_features_all_classes_train = cross_features_classI_train
		cross_features_all_classes_test = cross_features_classI_test
		cross_features_all_classes_labels_train = classI_labels_array_train
		cross_features_all_classes_labels_test = classI_labels_array_test
		cnt1 = 1
	else:
		cross_features_all_classes_train = np.vstack((cross_features_all_classes_train, cross_features_classI_train))
		cross_features_all_classes_test = np.vstack((cross_features_all_classes_test, cross_features_classI_test))
		cross_features_all_classes_labels_train = np.hstack((cross_features_all_classes_labels_train, classI_labels_array_train))	
		cross_features_all_classes_labels_test = np.hstack((cross_features_all_classes_labels_test, classI_labels_array_test))	

	print("there...........")
#	pdb.set_trace()

#pdb.set_trace()
obj_classifier_input = classifier_input()
obj_classifier_input.train_data = cross_features_all_classes_train
obj_classifier_input.number_of_train_classes = number_of_train_classes
obj_classifier_input.train_labels = cross_features_all_classes_labels_train
obj_classifier_input.test_data = cross_features_all_classes_test
obj_classifier_input.test_labels = cross_features_all_classes_labels_test
obj_classifier_output = function_train_classifier_for_cc(obj_classifier_input)

#pdb.set_trace()			
cc_end = time.time() 
