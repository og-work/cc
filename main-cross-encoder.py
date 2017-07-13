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
EPOCHS = 3
EPOCHS_CC = 500
BATCH_SIZE = 128
BATCH_SIZE_CC = 128
SYSTEM = 'desktop'; #desktop/laptop
TRAIN_VALIDATION_SPLIT = 0.8
MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES = 50

#Prepare encoder model...................
if SAMPLE_DATA:
	dimension_hidden_layer1 = 250
	dimension_hidden_layer2 = 100
	dimension_hidden_layer3 = 50
else:
	dimension_hidden_layer1 = 3
	dimension_hidden_layer2 = 2
	dimension_hidden_layer3 = 1
	

obj_input_data = input_data()
obj_input_data.dataset_name = 'apy'
obj_input_data.system_type = SYSTEM

obj_input_data = get_input_data(obj_input_data)
#pdb.set_trace()
unnormalised_data = obj_input_data.visual_features_dataset
normalised_data = normalise_data(unnormalised_data)
obj_input_data.visual_features_dataset = normalised_data
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
aec1.fit(train_samples, train_samples,
                epochs=EPOCHS,
                batch_size=256,
                shuffle=True,
                validation_data=(valid_samples, valid_samples),
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
aec_end = time.time()
aec1_time = aec_end - aec_start
print "aec training time: %f"%((aec_end - aec_start)/60) 

# encode and decode some digits
encoded_data_test1 = encoder1.predict(test_samples)
decoded_data_test1 = decoder1.predict(encoded_data_test1)
encoded_data_train1 = encoder1.predict(train_samples)
decoded_data_train1 = decoder1.predict(encoded_data_train1)
encoded_data_valid1 = encoder1.predict(valid_samples)
decoded_data_valid1 = decoder1.predict(encoded_data_valid1)

if 0:
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

#...................AEC 2..............
# this is our input placeholder
input_img2 = Input(shape=(dimension_hidden_layer1,))
# "encoded" is the encoded representation of the input
encoded2 = Dense(dimension_hidden_layer2, activation='relu')(input_img2)
# "decoded" is the lossy reconstruction of the input
decoded2 = Dense(dimension_hidden_layer1, activation='sigmoid')(encoded2)
# this model maps an input to its reconstruction
aec2 = Model(input_img2, decoded2)

# this model maps an input to its encoded representation
encoder2 = Model(input_img2, encoded2)
# create a placeholder for an encoded (32-dimensional) input
encoded_input2 = Input(shape=(dimension_hidden_layer2,))
# retrieve the last layer of the autoencoder model
decoder_layer2 = aec2.layers[-1]
# create the decoder model
decoder2 = Model(encoded_input2, decoder_layer2(encoded_input2))
aec2.compile(optimizer='adadelta', loss='binary_crossentropy')
aec_start = time.time()
#pdb.set_trace()
aec2.fit(encoded_data_train1, encoded_data_train1,
                epochs=EPOCHS,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_data_valid1, encoded_data_valid1),
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
aec_end = time.time()
print "aec training time: %f"%((aec_end - aec_start)/60) 
aec2_time = aec_end - aec_start

# encode and decode some digits
encoded_data_test2 = encoder2.predict(encoded_data_test1)
decoded_data_test2 = decoder2.predict(encoded_data_test2)
encoded_data_train2 = encoder2.predict(encoded_data_train1)
decoded_data_train2 = decoder2.predict(encoded_data_train2)
encoded_data_valid2 = encoder2.predict(encoded_data_valid1)
decoded_data_valid2 = decoder2.predict(encoded_data_valid2)

#Save data
scipy.io.savemat('data/aec2_encoded.mat', \
	dict(encoded_data_train2 = encoded_data_train2,\
	     encoded_data_valid2 = encoded_data_valid2, \
	     encoded_data_test2 = encoded_data_test2, \
	     decoded_data_train2 = decoded_data_train2, \
	     decoded_data_valid2 = decoded_data_valid2, \
	     decoded_data_test2 = decoded_data_test2, \
	     	))
print("Save encoded/decoded data for aec2.")

#Save model
#Serialize model to JSON
aec2_json = aec2.to_json()
with open("data/aec2.json", "w") as json_file:
    json_file.write(aec2_json)
# serialize weights to HDF5
aec2.save_weights("data/aec2.h5")
print("Saved model to disk")

#...................AEC 3..............
# this is our input placeholder
input_img3 = Input(shape=(dimension_hidden_layer2,))
# "encoded" is the encoded representation of the input
encoded3 = Dense(dimension_hidden_layer3, activation='relu')(input_img3)
# "decoded" is the lossy reconstruction of the input
decoded3 = Dense(dimension_hidden_layer2, activation='sigmoid')(encoded3)
# this model maps an input to its reconstruction
aec3 = Model(input_img3, decoded3)

# this model maps an input to its encoded representation
encoder3 = Model(input_img3, encoded3)
# create a placeholder for an encoded (32-dimensional) input
encoded_input3 = Input(shape=(dimension_hidden_layer3,))
# retrieve the last layer of the autoencoder model
decoder_layer3 = aec3.layers[-1]
# create the decoder model
decoder3 = Model(encoded_input3, decoder_layer3(encoded_input3))
aec3.compile(optimizer='adadelta', loss='binary_crossentropy')
aec_start = time.time()
aec3.fit(encoded_data_train2, encoded_data_train2,
                epochs=EPOCHS,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_data_valid2, encoded_data_valid2),
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
aec_end = time.time()
print "aec training time: %f"%((aec_end - aec_start)/60) 
aec3_time = aec_end - aec_start

# encode and decode some digits
encoded_data_test3 = encoder3.predict(encoded_data_test2)
decoded_data_test3 = decoder3.predict(encoded_data_test3)
encoded_data_train3 = encoder3.predict(encoded_data_train2)
decoded_data_train3 = decoder3.predict(encoded_data_train3)
encoded_data_valid3 = encoder3.predict(encoded_data_valid2)
decoded_data_valid3 = decoder3.predict(encoded_data_valid3)

#Save data
scipy.io.savemat('data/aec3_encoded.mat', \
	dict(encoded_data_train3 = encoded_data_train3,\
	     encoded_data_valid3 = encoded_data_valid3, \
	     encoded_data_test3 = encoded_data_test3, \
	     decoded_data_train3 = decoded_data_train3, \
	     decoded_data_valid3 = decoded_data_valid3, \
	     decoded_data_test3 = decoded_data_test3, \
	     	))
print("Save encoded/decoded data for aec3.")

#Save model
#Serialize model to JSON
aec3_json = aec3.to_json()
with open("data/aec3.json", "w") as json_file:
    json_file.write(aec3_json)
# serialize weights to HDF5
aec3.save_weights("data/aec3.h5")
print("Saved model to disk")

pdb.set_trace()
#......................cc.......................
number_of_cc = number_of_train_classes * number_of_train_classes - number_of_train_classes

cross_coders_train_data_input = []
cross_coders_train_data_output = []
auto_cc_features_train = decoded_data_train3
auto_cc_features_test = decoded_data_test3
auto_cc_features_valid = decoded_data_valid3

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
			encoded_cc1 = Dense(encoding_dimension_cc1, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_cc1)
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
			cc1_start = time.time()
			cc1.fit(cc1_train_valid_data.input_train, cc1_train_valid_data.output_train,
            			    epochs=EPOCHS_CC,
			                batch_size=BATCH_SIZE_CC,
			                shuffle=True,
			                validation_data=(cc1_train_valid_data.input_valid, cc1_train_valid_data.output_valid))
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
			encoded_data_train_cc1 = encoder_cc1.predict(cc1_train_valid_data.input_train)	
			encoded_data_valid_cc1 = encoder_cc1.predict(cc1_train_valid_data.input_valid)	
			
			file_name = 'data/cc1_encoded_' + str(classI) + '_' + str(classJ) + '.mat'		
			scipy.io.savemat(file_name, \
				dict(encoded_data_train_cc1 = encoded_data_train_cc1,\
				     encoded_data_valid_cc1 = encoded_data_valid_cc1,))
			print "Save encoded/decoded data for cc1: %s" %file_name
	 
			#..................cc2.........................
			encoding_dimension_cc2 = dimension_hidden_layer2
			input_cc2 = Input(shape=(dimension_hidden_layer1,))
			encoded_cc2 = Dense(encoding_dimension_cc2, activation='relu')(input_cc2)
			decoded_cc2 = Dense(dimension_hidden_layer1, activation='sigmoid')(encoded_cc2)
			cc2 = Model(input_cc2, decoded_cc2)
			encoder_cc2 = Model(input_cc2, encoded_cc2)
			encoded_input_cc2 = Input(shape=(encoding_dimension_cc2,))
			decoder_layer_cc2 = cc2.layers[-1]
			decoder_cc2 = Model(encoded_input_cc2, decoder_layer_cc2(encoded_input_cc2))
			cc2.compile(optimizer='adadelta', loss='binary_crossentropy')
			cc2_start = time.time()
			cc2.fit(encoded_data_train_cc1, encoded_data_train_cc1,
            			    epochs=EPOCHS_CC,
			                batch_size=BATCH_SIZE_CC,
			                shuffle=True,
			                validation_data=(encoded_data_valid_cc1, encoded_data_valid_cc1))
			cc2_end = time.time()
			cc2_time = cc2_end - cc2_start

			#Save models to JSON
			modelName = 'data/cc2_' + str(classI) + '_' + str(classJ) + '.json' 
			cc2_json = cc2.to_json()
			with open(modelName, "w") as json_file:
			    json_file.write(cc2_json)
			# serialize weights to HDF5
			modelNameh5 = 'data/cc2_' + str(classI) + '_' + str(classJ) + '.h5'
			cc2.save_weights(modelNameh5)
			
			#Get cc features for training samples
			encoded_data_train_cc2 = encoder_cc2.predict(encoded_data_train_cc1)	
			encoded_data_valid_cc2 = encoder_cc2.predict(encoded_data_valid_cc1)	

			file_name = 'data/cc2_encoded_' + str(classI) + '_' + str(classJ) + '.mat'		
			scipy.io.savemat(file_name, \
				dict(encoded_data_train_cc2 = encoded_data_train_cc2,\
				     encoded_data_valid_cc2 = encoded_data_valid_cc2,))
			print "Saved encoded/decoded data for cc2: %s" %file_name

			#...................cc3........................
			encoding_dimension_cc3 = dimension_hidden_layer3
			input_cc3 = Input(shape=(dimension_hidden_layer2,))
			encoded_cc3 = Dense(encoding_dimension_cc3, activation='relu')(input_cc3)
			decoded_cc3 = Dense(dimension_hidden_layer2, activation='sigmoid')(encoded_cc3)
			cc3 = Model(input_cc3, decoded_cc3)
			encoder_cc3 = Model(input_cc3, encoded_cc3)
			encoded_input_cc3 = Input(shape=(encoding_dimension_cc3,))
			decoder_layer_cc3 = cc3.layers[-1]
			decoder_cc3 = Model(encoded_input_cc3, decoder_layer_cc3(encoded_input_cc3))
			cc3.compile(optimizer='adadelta', loss='binary_crossentropy')
			cc3_start = time.time()	
			cc3.fit(encoded_data_train_cc2, encoded_data_train_cc2,
            			    epochs=EPOCHS_CC,
			                batch_size=BATCH_SIZE_CC,
			                shuffle=True,
			                validation_data=(encoded_data_valid_cc2, encoded_data_valid_cc2))
			cc3_end = time.time()
			cc3_time = cc3_end - cc3_start

			#Save models to JSON
			modelName = 'data/cc3_' + str(classI) + '_' + str(classJ) + '.json' 
			cc3_json = cc3.to_json()
			with open(modelName, "w") as json_file:
			    json_file.write(cc3_json)
			# serialize weights to HDF5
			modelNameh5 = 'data/cc3_' + str(classI) + '_' + str(classJ) + '.h5'
			cc3.save_weights(modelNameh5)
			
			#Get cc features for training samples
			encoded_data_train_cc3 = encoder_cc3.predict(encoded_data_train_cc2)	
			encoded_data_valid_cc3 = encoder_cc3.predict(encoded_data_valid_cc2)	
			
			file_name = 'data/cc3_encoded_' + str(classI) + '_' + str(classJ) + '.mat'		
			scipy.io.savemat(file_name, \
				dict(encoded_data_train_cc3 = encoded_data_train_cc3,\
				     encoded_data_valid_cc3 = encoded_data_valid_cc3,))
			print "Saved encoded/decoded data for cc3: %s" %file_name
			#stack features.....................................
			#auto_cc_features_train = np.append(auto_cc_features_train, decoded_data_train_cc, axis = 1)
			#auto_cc_features_valid = np.append(auto_cc_features_valid, decoded_data_valid_cc, axis = 1)
			#auto_cc_features_test = np.append(auto_cc_features_test, decoded_data_test_cc, axis = 1)

cc_end = time.time() 
total_aec_time = aec1_time + aec2_time + aec3_time
total_cc_time = cc1_time + cc2_time + cc3_time
total_time = total_aec_time + total_cc_time
print "Total processing time %f"%(total_time/3600)
print
