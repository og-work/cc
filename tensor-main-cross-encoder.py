from get_data_for_cc import function_get_training_data_cc, input_cc, output_cc, normalise_data
from get_data_for_cc import input_data, get_input_data
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import scipy.io
import matplotlib
from keras.datasets import mnist
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import pdb
print "*****************************************************************************************************************************************"

SAMPLE_DATA = 1
EPOCHS = 10000
EPOCHS_CC = 500
BATCH_SIZE = 128
BATCH_SIZE_CC = 128
SYSTEM = 'desktop'; #desktop/laptop
TRAIN_VALIDATION_SPLIT = 0.8
MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES = 50

#Prepare encoder model...................
if SAMPLE_DATA:
	dimension_hidden_layer1 = 50
	dimension_hidden_layer2 = 100
	dimension_hidden_layer3 = 50
else:
	dimension_hidden_layer1 = 3
	dimension_hidden_layer2 = 2
	dimension_hidden_layer3 = 1
	

obj_input_data = input_data()
obj_input_data.dataset_name = 'sample'#apy
obj_input_data.system_type = SYSTEM

obj_input_data = get_input_data(obj_input_data)
#pdb.set_trace()
visual_features_dataset = obj_input_data.visual_features_dataset
train_class_labels = obj_input_data.train_class_labels
test_class_labels = obj_input_data.test_class_labels
attributes = obj_input_data.attributes
dataset_labels = obj_input_data.dataset_labels

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
print(train_class_labels)
print(test_class_labels)

#Get training and validation samples
train_sample_indices = np.array([])
train_sample_labels = np.array([])
valid_sample_indices = np.array([])
valid_sample_labels = np.array([])
number_of_samples_per_class_train = []
number_of_samples_per_class_valid = []

#pdb.set_trace()
for class_index in train_class_labels: 
	indices = np.flatnonzero(dataset_labels == class_index)
	#number_of_samples_for_train = int(TRAIN_VALIDATION_SPLIT * MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES)
	number_of_samples_for_train = int(TRAIN_VALIDATION_SPLIT * np.size(indices))
	indices_train = indices[:number_of_samples_for_train]
	#indices_valid = indices[number_of_samples_for_train:MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES]
	indices_valid = indices[number_of_samples_for_train:]
	number_of_samples_per_class_train.append(number_of_samples_for_train)
	number_of_samples_per_class_valid.append(np.size(indices) - number_of_samples_for_train)
	train_sample_indices = np.concatenate((train_sample_indices, indices_train), axis = 0)
	valid_sample_indices = np.concatenate((valid_sample_indices, indices_valid), axis = 0)
	train_labels = np.empty(number_of_samples_for_train)
	train_labels.fill(class_index)
	train_sample_labels =  np.concatenate((train_sample_labels, train_labels), axis = 0)
	valid_labels = np.empty(np.size(indices_valid))
	valid_labels.fill(class_index)
	valid_sample_labels =  np.concatenate((valid_sample_labels, valid_labels), axis = 0)

train_samples = visual_features_dataset[train_sample_indices.astype(int), :]
valid_samples = visual_features_dataset[valid_sample_indices.astype(int), :]

#increase nummber of samples by adding random noise
NOISE_FACTOR = 0
INCREASE_FACTOR = 1
train_samples_without_noise = np.tile(train_samples, (INCREASE_FACTOR, 1))
train_samples_noisy = train_samples_without_noise + NOISE_FACTOR * np.random.normal(0, 1, train_samples_without_noise.shape)

train_samples_noisy = normalise_data(train_samples_noisy)
train_samples_without_noise = normalise_data(train_samples_without_noise)
valid_samples = normalise_data(valid_samples)

#Get testing samples
test_sample_indices = np.array([])
test_sample_labels = np.array([])

for class_index in test_class_labels: 
	indices = np.flatnonzero(dataset_labels == class_index)
	test_sample_indices = np.concatenate((test_sample_indices, indices), axis = 0)
	test_labels = np.empty(np.size(indices))
	test_labels.fill(class_index)
	test_sample_labels =  np.concatenate((test_sample_labels, test_labels), axis = 0)

test_samples = visual_features_dataset[test_sample_indices.astype(int), :]

print "Train samples of size: %d X %d" %train_samples.shape
print "Test samples of size: %d X %d" %test_samples.shape

pdb.set_trace()

if SAMPLE_DATA:
	for k in range(0, train_samples.shape[0]):
		print(train_samples[k, :])
	print('*******')
	for k in range(0, test_samples.shape[0]):
		print(test_samples[k, :])
	print(train_sample_indices.shape)
	print(train_sample_indices)
	print(test_sample_indices)


#...............stacked AECs...........
#...................AEC 1..............
# this is our input placeholder
input_img1 = Input(shape=(dimension_visual_data,))
# "encoded" is the encoded representation of the input
encoded1 = Dense(dimension_hidden_layer1, activation='relu')(input_img1)
# "decoded" is the lossy reconstruction of the input
decoded1 = Dense(dimension_visual_data, activation='sigmoid')(encoded1)
# this model maps an input to its reconstruction
aec1 = Model(input_img1, decoded1)

# this model maps an input to its encoded representation
encoder1 = Model(input_img1, encoded1)
# create a placeholder for an encoded (32-dimensional) input
encoded_input1 = Input(shape=(dimension_hidden_layer1,))
# retrieve the last layer of the autoencoder model
decoder_layer1 = aec1.layers[-1]
# create the decoder model
decoder1 = Model(encoded_input1, decoder_layer1(encoded_input1))
aec1.compile(optimizer='adadelta', loss='binary_crossentropy')
aec_start = time.time()
aec1.fit(train_samples_noisy, train_samples_without_noise,
                epochs=EPOCHS,
                batch_size=256,
                shuffle=True,
                validation_data=(valid_samples, valid_samples),
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
aec_end = time.time()
aec1_time = aec_end - aec_start
print "aec training time: %f"%((aec_end - aec_start)/60) 

# encode and decode some digits
test_samples = normalise_data(test_samples)
encoded_data_test1 = encoder1.predict(test_samples)
decoded_data_test1 = decoder1.predict(encoded_data_test1)
train_samples = normalise_data(train_samples)
encoded_data_train1 = encoder1.predict(train_samples)
decoded_data_train1 = decoder1.predict(encoded_data_train1)
valid_samples = normalise_data(valid_samples)
encoded_data_valid1 = encoder1.predict(valid_samples)
decoded_data_valid1 = decoder1.predict(encoded_data_valid1)

if 1:
	scipy.io.savemat('data/aec1_encoded.mat', \
	dict(encoded_data_train1 = encoded_data_train1,\
	     encoded_data_valid1 = encoded_data_valid1, \
	     encoded_data_test1 = encoded_data_test1, \
	     decoded_data_train1 = decoded_data_train1, \
	     decoded_data_valid1 = decoded_data_valid1, \
	     decoded_data_test1 = decoded_data_test1, \
	     aec1_input_train = train_samples,\
	     aec1_input_valid = valid_samples,\
	     aec1_input_labels_train = train_sample_labels,\
	     aec1_input_labels_valid = valid_sample_labels,\
	     	))
	print("Save encoded/decoded data for aec1.")

#Save model
#Serialize model to JSON
aec1_json = aec1.to_json()
with open("data/aec1.json", "w") as json_file:
    json_file.write(aec1_json)
# serialize weights to HDF5
aec1.save_weights("data/aec1.h5")
print("Saved model to disk")

#Code snnippet to get layer weights
#Manually....
for layer in aec1.layers:
    weights = layer.get_weights() # list of numpy arrays

pdb.set_trace()
for layer in aec1.layers:
	h = layer.get_weights()
	print(h)

#......................cc.......................
number_of_cc = number_of_train_classes * number_of_train_classes - number_of_train_classes

cross_coders_train_data_input = []
cross_coders_train_data_output = []

#Get mean feature vector for each class
mean_feature_mat = np.empty((0, dimension_visual_data), float)
for classI in train_class_labels:
	ind = np.flatnonzero(dataset_labels == classI)
	classI_features = visual_features_dataset[ind.astype(int), :]
	mean_feature = classI_features.mean(0)
	mean_feature_mat = np.append(mean_feature_mat, mean_feature.reshape(1, dimension_visual_data), axis = 0)	


cc_start = time.time() 
for classI in train_class_labels:
	for classJ in train_class_labels:
		print "**************************************"
		if classI != classJ:
			
			#..................cc1.........................
			encoding_dimension_cc1 = dimension_hidden_layer1
			input_cc1 = Input(shape=(dimension_visual_data,))
			encoded_cc1 = Dense(encoding_dimension_cc1, activation='relu')(input_cc1)
			decoded_cc1 = Dense(dimension_visual_data, activation='sigmoid')(encoded_cc1)
			cc1 = Model(input_cc1, decoded_cc1)
			encoder_cc1 = Model(input_cc1, encoded_cc1)
			encoded_input_cc1 = Input(shape=(encoding_dimension_cc1,))
			decoder_layer_cc1 = cc1.layers[-1]
			decoder_cc1 = Model(encoded_input_cc1, decoder_layer_cc1(encoded_input_cc1))
			cc1.compile(optimizer='adadelta', loss='binary_crossentropy')
			
			obj_input_cc = input_cc()
			obj_input_cc.classI = classI
			obj_input_cc.classJ = classJ
			obj_input_cc.visual_features = visual_features_dataset
			obj_input_cc.train_valid_split = TRAIN_VALIDATION_SPLIT
			obj_input_cc.dataset_labels = dataset_labels
			
			cc1_train_valid_data = function_get_training_data_cc(obj_input_cc)
			cc1_input_train = cc1_train_valid_data.input_train
			cc1_input_train = np.tile(cc1_input_train, (INCREASE_FACTOR, 1))
			cc1_input_train = cc1_input_train + NOISE_FACTOR * np.random.normal(0, 1, cc1_input_train.shape)
			cc1_output_train = cc1_train_valid_data.output_train
			cc1_output_train = np.tile(cc1_output_train, (INCREASE_FACTOR, 1))
			cc1_output_train = cc1_output_train + NOISE_FACTOR * np.random.normal(0, 1, cc1_output_train.shape)
			cc1_input_train = normalise_data(cc1_input_train)
			cc1_output_train = normalise_data(cc1_output_train)
			cc1_start = time.time()
			cc1.fit(cc1_input_train, cc1_output_train,
            			    epochs=EPOCHS_CC,
			                batch_size=BATCH_SIZE_CC,
			                shuffle=True,
			                validation_data=(normalise_data(cc1_train_valid_data.input_valid), \
											normalise_data( cc1_train_valid_data.output_valid)))
			cc1_end = time.time()
			cc1_time = cc1_end - cc1_start
			#Save models to JSON
			modelName = 'data/cc1_' + str(classI) + '_' + str(classJ) + '.json' 
			cc1_json = cc1.to_json()
			with open(modelName, "w") as json_file:
			    json_file.write(cc1_json)
			# serialize weights to HDF5
			modelNameh5 = 'data/cc1_' + str(classI) + '_' + str(classJ) + '.h5'
			cc1.save_weights(modelNameh5)
			
			#Get cc features for training samples
			encoded_data_train_cc1 = encoder_cc1.predict(normalise_data(cc1_train_valid_data.input_train))	
			decoded_data_train_cc1 = decoder_cc1.predict(encoded_data_train_cc1)	
			encoded_data_valid_cc1 = encoder_cc1.predict(cc1_train_valid_data.input_valid)	
			
			file_name = 'data/cc1_encoded_' + str(classI) + '_' + str(classJ) + '.mat'		
			scipy.io.savemat(file_name, \
				dict(encoded_data_train_cc1 = encoded_data_train_cc1,\
				     encoded_data_valid_cc1 = encoded_data_valid_cc1,))
			print "Save encoded/decoded data for cc1: %s" %file_name

cc_end = time.time() 
total_aec_time = aec1_time + aec2_time + aec3_time
total_cc_time = cc1_time + cc2_time + cc3_time
total_time = total_aec_time + total_cc_time
print "Total processing time %f"%(total_time/3600)
print
