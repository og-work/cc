import numpy as np
import pdb
import scipy.io
import tensorflow as tf
from keras.callbacks import TensorBoard

class output_cc:
	input_train_perm = np.array([])
	output_train_perm = np.array([])
	input_valid_perm = np.array([])
	output_valid_perm = np.array([])
	input_test_perm = np.array([])

	input_train = np.array([])
	input_valid = np.array([])
	input_test = np.array([])
	output_test = np.array([])
	
	indices_input_samples_train_perm = np.array([])
	indices_ouput_samples_train_perm = np.array([])
	indices_input_samples_test_perm = np.array([])
	
	indices_classI_samples_train = np.array([])
	indices_classI_samples_test = np.array([])
	indices_classJ_samples_test = np.array([])
	indices_classI_samples_valid = np.array([])

	def function(self):
		print("This is output_cc class")

class input_cc:
	classI = []
	classJ = []
	train_valid_split = np.array([])
	visual_features = np.array([])
	dataset_labels = np.array([])

	def function(self):
		print("This is input_cc class")

def function_get_training_data_cc(obj_input_cc):
	#pdb.set_trace()
	classI = obj_input_cc.classI
	classJ = obj_input_cc.classJ
	TR_TS_VA_SPLIT = obj_input_cc.train_valid_split
	dataset_labels = obj_input_cc.dataset_labels
	visual_features_dataset = obj_input_cc.visual_features

	print "************* Class %d and Class %d*************************"%(classI, classJ)
	MAX_NUMBER_OF_SAMPLES_CLASSI_TRAIN = 30 # for apy class 1-32: 30
	MAX_NUMBER_OF_SAMPLES_CLASSJ_TRAIN = 30 # for apy class 1-32: 30

	if classI != classJ:
		indices_classI_samples = np.flatnonzero(dataset_labels == classI)
		indices_classJ_samples = np.flatnonzero(dataset_labels == classJ)

		#number_of_samples_classI_for_train = int(TR_TS_VA_SPLIT[0] * np.size(indices_classI_samples))
		#number_of_samples_classJ_for_train = int(TR_TS_VA_SPLIT[0] * np.size(indices_classJ_samples))
		
		number_of_samples_classI_for_train = MAX_NUMBER_OF_SAMPLES_CLASSI_TRAIN
		number_of_samples_classJ_for_train = MAX_NUMBER_OF_SAMPLES_CLASSJ_TRAIN
		
		indices_classI_samples_train = indices_classI_samples[:number_of_samples_classI_for_train]
		indices_classJ_samples_train = indices_classJ_samples[:number_of_samples_classJ_for_train]
		
		number_of_samples_classI_for_valid = int(TR_TS_VA_SPLIT[1] * np.size(indices_classI_samples))
		number_of_samples_classJ_for_valid = int(TR_TS_VA_SPLIT[1] * np.size(indices_classJ_samples))

		number_of_samples_classI_for_test = int(TR_TS_VA_SPLIT[2] * np.size(indices_classI_samples))
		number_of_samples_classJ_for_test = number_of_samples_classI_for_test #imp
		
		start_vl_classI = number_of_samples_classI_for_train
		end_vl_classI = start_vl_classI + number_of_samples_classI_for_valid

		start_vl_classJ = number_of_samples_classJ_for_train
		end_vl_classJ = start_vl_classJ + number_of_samples_classJ_for_valid
		
		start_ts_classI = end_vl_classI
		end_ts_classI = start_ts_classI + number_of_samples_classI_for_test	
		
		start_ts_classJ = end_vl_classJ
		end_ts_classJ = start_ts_classJ + number_of_samples_classJ_for_test	

		indices_classI_samples_valid = indices_classI_samples[start_vl_classI:end_vl_classI]
		indices_classJ_samples_valid = indices_classJ_samples[start_vl_classJ:end_vl_classJ]

		indices_classI_samples_test = indices_classI_samples[start_ts_classI:end_ts_classI]
		indices_classJ_samples_test = indices_classJ_samples[start_ts_classJ:end_ts_classJ]
		
		print "Total class %d samples %d "%(classI, np.size(indices_classI_samples))
	        print "data split tr %d valid %d ts %d" %(number_of_samples_classI_for_train, \
	            number_of_samples_classI_for_valid, number_of_samples_classI_for_test)
	        print "Total class %d samples %d "%(classJ, np.size(indices_classJ_samples))
        	print "data split tr %d val %d" %(number_of_samples_classJ_for_train, \
	            number_of_samples_classJ_for_valid)
		#print(indices_classI_samples)
		#print(indices_classI_samplesTrain)
		#print(indices_classI_samplesValid)
		#print(indices_classJ_samples)
		#print(indices_classJ_samplesTrain)
		#print(indices_classJ_samplesValid)
	
		#Prepare train data for cc
		indices_input_sample_train = np.array([])
		indices_output_sample_train = np.array([])
	
		for index_classI_sample in indices_classI_samples_train:
			indices_classI_sample_array = np.empty(indices_classJ_samples_train.size)
			indices_classI_sample_array.fill(index_classI_sample)
			indices_input_sample_train = np.concatenate((indices_input_sample_train, indices_classI_sample_array), axis = 0)
			indices_output_sample_train = np.concatenate((indices_output_sample_train, indices_classJ_samples_train), axis = 0)

		#pdb.set_trace()
		#Prepare validation data for cc
		indices_input_sample_valid = np.array([])
		indices_output_sample_valid = np.array([])
	
		for index_classI_sample in indices_classI_samples_valid:
			indices_classI_sample_array = np.empty(indices_classJ_samples_valid.size)
			indices_classI_sample_array.fill(index_classI_sample)
			indices_input_sample_valid = np.concatenate((indices_input_sample_valid, indices_classI_sample_array), axis = 0)
			indices_output_sample_valid = np.concatenate((indices_output_sample_valid, indices_classJ_samples_valid), axis = 0)
	
		#pdb.set_trace()
		print "Number of samples for CC train %d, for validation %d" %(np.size(indices_input_sample_train), np.size(indices_input_sample_valid))			
	
		#unit test	
		if (np.size(indices_input_sample_train) != np.size(indices_output_sample_train)):
			raise NameError('Input and output data dimensions are not matching for CC')

		obj_output_cc = output_cc()
		#permuted data
		obj_output_cc.input_train_perm = visual_features_dataset[indices_input_sample_train.astype(int), :]
		obj_output_cc.output_train_perm = visual_features_dataset[indices_output_sample_train.astype(int), :]
		obj_output_cc.input_valid_perm = visual_features_dataset[indices_input_sample_valid.astype(int), :]
		obj_output_cc.output_valid_perm = visual_features_dataset[indices_output_sample_valid.astype(int), :]

		#permuted indices
		obj_output_cc.indices_input_samples_train_perm = indices_input_sample_train
		obj_output_cc.indices_ouput_samples_train_perm = indices_output_sample_train
		obj_output_cc.indices_input_samples_test_perm = indices_classI_samples_test

		#Non-permuted data		
		obj_output_cc.input_train = visual_features_dataset[indices_classI_samples_train.astype(int), :]
		obj_output_cc.input_test = visual_features_dataset[indices_classI_samples_test.astype(int), :] 
		obj_output_cc.output_test = visual_features_dataset[indices_classJ_samples_test.astype(int), :] 
		obj_output_cc.input_valid = visual_features_dataset[indices_classI_samples_valid.astype(int), :]
		
		#Non-permuted indices		
		obj_output_cc.indices_classI_samples_train = indices_classI_samples_train
		obj_output_cc.indices_classI_samples_valid = indices_classI_samples_valid
		obj_output_cc.indices_classI_samples_test = indices_classI_samples_test
		obj_output_cc.indices_classJ_samples_test = indices_classJ_samples_test
		#pdb.set_trace()
		return obj_output_cc

class input_data:
	dataset_name = []
	system_type = []
	train_class_labels = np.array([])
	test_class_labels = np.array([])
	visual_features_dataset = np.array([])
	attributes = np.array([])
	dataset_labels = np.array([])
	def function(self):
		print("This is the input_data class")	

def function_normalise_data(unnormalised_data):
	#Normalise along each dimension separately
	#max_val_array = unnormalised_data.max(axis = 0)
	#max_val_array[max_val_array == 0] = 1.
	#max_val_mat = np.tile(max_val_array, (unnormalised_data.shape[0], 1))
	#normalised_data = unnormalised_data/max_val_mat
	
	#Normalise entire data between [0,1]
	normalised_data = np.divide((unnormalised_data - unnormalised_data.min()), (unnormalised_data.max() - unnormalised_data.min()))
	return normalised_data

def function_get_input_data(obj_input_data):

	if obj_input_data.system_type == 'desktop':
		BASE_PATH = "/nfs4/omkar/Documents/"
	else:
		BASE_PATH = "/media/omkar/windows-D-drive/"
	print(BASE_PATH)
	if obj_input_data.dataset_name == 'apy':
		path_CNN_features = BASE_PATH + "/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/cnn_feat_imagenet-vgg-verydeep-19.mat"
		path_attributes = BASE_PATH + "/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/class_attributes.mat"
		features = scipy.io.loadmat(path_CNN_features)
		attributes_data = scipy.io.loadmat(path_attributes)
		attributes = attributes_data['class_attributes']
		dataset_labels = attributes_data['labels']
		visual_features_dataset = features['cnn_feat']
		visual_features_dataset = visual_features_dataset.transpose()
		train_class_labels = np.array([1, 2, 3, 4, 5])
		#train_class_labels = np.arange(1, 33, 1)
		test_class_labels = np.arange(21, 33, 1)
        
#'1 aeroplane' '2 bicycle''3 bird''4 boat''5 bottle''6 bus''7 car''8 cat''9 chair''10 cow''11 diningtable''12 dog''13 horse'
	    #'14 motorbike''15 person''16 pottedplant''17 sheep''18 sofa''19 train''20 tvmonitor''21 donkey''22 monkey''23 goat''24 wolf'
    	#'25 jetski''26 zebra''27 centaur''28 mug''29 statue''30 building''31 bag''32 carriage'
	else:
		dataset_labels = np.array([1, 1, 1, 1, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3])
		attributes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
		visual_features_dataset = np.array([[11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
	                                      [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
                                      [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
                                      [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22.0, 33],
                                      [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33.0]], dtype='f')
		visual_features_dataset = visual_features_dataset.transpose()
		train_class_labels = np.array([1, 2, 3])
		test_class_labels = np.array([1	, 2, 3])
		
	obj_input_data.test_class_labels = test_class_labels 
	obj_input_data.train_class_labels = train_class_labels 
	obj_input_data.attributes = attributes 
	obj_input_data.visual_features_dataset = visual_features_dataset
	obj_input_data.dataset_labels = dataset_labels

	return obj_input_data
