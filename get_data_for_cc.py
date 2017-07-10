import numpy as np
import pdb

class output_cc:
	input_train = np.array([])
	output_train = np.array([])
	input_valid = np.array([])
	output_valid = np.array([])

	def function(self):
		print("This is output_cc class")

class input_cc:
	classI = []
	classJ = []
	train_valid_split = []
	visual_features = np.array([])
	dataset_labels = np.array([])

	def function(self):
		print("This is input_cc class")

def function_get_training_data_cc():

	indices_input_sample_train = np.array([])
	indices_output_sample_train = np.array([])
	indices_input_sample_valid = np.array([])
	indices_output_sample_valid = np.array([])
	print "**************************************"
	
	if classI != classJ:
		indices_classI_samples = np.flatnonzero(dataset_labels == classI)
		indices_classJ_samples = np.flatnonzero(dataset_labels == classJ)
		number_of_samples_classI_for_train = int(TRAIN_VALIDATION_SPLIT * np.size(indices_classI_samples))
		number_of_samples_classJ_for_train = int(TRAIN_VALIDATION_SPLIT * np.size(indices_classJ_samples))
		indices_classI_samples_train = indices_classI_samples[:number_of_samples_classI_for_train]
		indices_classI_samples_valid = indices_classI_samples[number_of_samples_classI_for_train:]
		indices_classJ_samples_train = indices_classJ_samples[:number_of_samples_classJ_for_train]
		indices_classJ_samples_valid = indices_classJ_samples[number_of_samples_classJ_for_train:]
		print "classI %d classJ %d indices %d %d %d %d" %(classI, classJ, indices_classI_samples.size, indices_classJ_samples.size, \
				number_of_samples_classI_for_train, number_of_samples_classJ_for_train)
		#print(indices_classI_samples)
		#print(indices_classI_samplesTrain)
		#print(indices_classI_samplesValid)
		#print(indices_classJ_samples)
		#print(indices_classJ_samplesTrain)
		#print(indices_classJ_samplesValid)
	
		#Prepare train data for CC
		for index_classI_sample in indices_classI_samples_train:
			indices_classI_sample_array = np.empty(indices_classJ_samples_train.size)
			indices_classI_sample_array.fill(index_classI_sample)
			indices_input_sample_train = np.concatenate((indices_input_sample_train, indices_classI_sample_array), axis = 0)
			indices_output_sample_train = np.concatenate((indices_output_sample_train, indices_classJ_samples_train), axis = 0)

		#Prepare validation data for CC
		for index_classI_sample in indices_classI_samples_valid:
			indices_classI_sample_array = np.empty(indices_classJ_samples_valid.size)
			indices_classI_sample_array.fill(index_classI_sample)
			indices_input_sample_valid = np.concatenate((indices_input_sample_valid, indices_classI_sample_array), axis = 0)
			indices_output_sample_valid = np.concatenate((indices_output_sample_valid, indices_classJ_samples_valid), axis = 0)
			#print(indices_input_sample_train)							
			#print(indices_output_sample_train)
			#crossCodersTrainDataInput.append(indices_input_sample_train)							
			#crossCodersTrainDataOutput.append(indices_output_sample_train)							
			
		input_samples_for_CC_train = visual_features_dataset[indices_input_sample_train.astype(int), :]
		output_samples_for_CC_train = visual_features_dataset[indices_output_sample_train.astype(int), :]
		input_samples_for_CC_valid = visual_features_dataset[indices_input_sample_valid.astype(int), :]
		outpt_samples_for_CC_valid = visual_features_dataset[indices_output_sample_valid.astype(int), :]
		
		obj_output_cc = output_cc()
		obj_output_cc.input_train = input_samples_for_CC_train
		obj_output_cc.output_train = output_samples_for_CC_train
		obj_output_cc.input_valid = input_samples_for_CC_valid
		obj_output_cc.output_valid = output_samples_for_CC_valid

		return obj_output_cc
