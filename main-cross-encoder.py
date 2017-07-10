from test_function_file import function_get_training_data_cc, input_cc, output_cc
from keras.layers import Input, Dense
from keras.models import Model
import scipy.io
import matplotlib
from keras.datasets import mnist
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import pdb
print "*****************************************************************************************************************************************"

SAMPLE_DATA = 1
EPOCHS = 1
BATCH_SIZE = 256
USE_DESKTOP = 0;
TRAIN_VALIDATION_SPLIT = 0.5

if USE_DESKTOP:
	BASE_PATH = '/nfs4/omkar/Documents/'
else:
	BASE_PATH = '/media/omkar/windows-D-drive/'

if 0 == SAMPLE_DATA:
	path_CNN_features = BASE_PATH + "/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/cnn_feat_imagenet-vgg-verydeep-19.mat"
	path_attributes = BASE_PATH + "/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/class_attributes.mat"
	features = scipy.io.loadmat(path_CNN_features)
	attributes_data = scipy.io.loadmat(path_attributes)
	attributes = attributes_data['class_attributes']
	dataset_labels = attributes_data['labels']
	visual_features_dataset = features['cnn_feat']
	visual_features_dataset = visual_features_dataset.transpose()
	train_class_labels = np.arange(1, 33, 1)
	test_class_labels = np.arange(21, 33, 1)
else:
	dataset_labels = np.array([1, 1, 1, 1, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3])
	attributes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
	visual_features_dataset = np.array([[11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33]])
	visual_features_dataset = visual_features_dataset.transpose()
	train_class_labels = np.array([1, 3])
	test_class_labels = np.array([2, 3])

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

for class_index in train_class_labels: 
	indices = np.flatnonzero(dataset_labels == class_index)
	#number_of_samples_for_train = TRAIN_VALIDATION_SPLIT * MIN_NUMBER_OF_SAMPLES_ACROSS_CLASSES
	number_of_samples_for_train = int(TRAIN_VALIDATION_SPLIT * np.size(indices))
	indices_train = indices[:number_of_samples_for_train]
	indices_valid = indices[number_of_samples_for_train:]
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

#Prepare encoder model...................
dimension_hidden_layer1 = 1000
dimension_hidden_layer2 = 500
dimension_hidden_layer3 = 300

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
print "aec training time: %f"%((aec_end - aec_start)/60) 

# encode and decode some digits
encoded_data_test1 = encoder1.predict(test_samples)
decoded_data_test1 = decoder1.predict(encoded_data_test1)
encoded_data_train1 = encoder1.predict(train_samples)
decoded_data_train1 = decoder1.predict(encoded_data_train1)
encoded_data_valid1 = encoder1.predict(valid_samples)
decoded_data_valid1 = decoder1.predict(encoded_data_valid1)

scipy.io.savemat('data/aec1_encoded.mat', \
	dict(encoded_data_train1 = encoded_data_train1, encoded_data_valid1 = encoded_data_valid1))
pdb.set_trace()

#Save model
#Serialize model to JSON
aec1_json = aec1.to_json()
with open("data/aec1.json", "w") as json_file:
    json_file.write(aec1_json)
# serialize weights to HDF5
aec1.save_weights("data/aec1.h5")
print("Saved model to disk")

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
aec2.fit(encoded_data_train1, encoded_data_train1,
                epochs=EPOCHS,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_data_valid1, encoded_data_valid1),
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
aec_end = time.time()
print "aec training time: %f"%((aec_end - aec_start)/60) 

# encode and decode some digits
encoded_data_test2 = encoder2.predict(encoded_data_test1)
decoded_data_test2 = decoder2.predict(encoded_data_test2)
encoded_data_train2 = encoder2.predict(encoded_data_train1)
decoded_data_train2 = decoder2.predict(encoded_data_train2)
encoded_data_valid2 = encoder2.predict(encoded_data_valid1)
decoded_data_valid2 = decoder2.predict(encoded_data_valid2)

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

# encode and decode some digits
encoded_data_test3 = encoder3.predict(encoded_data_test2)
decoded_data_test3 = decoder3.predict(encoded_data_test3)
encoded_data_train3 = encoder3.predict(encoded_data_train2)
decoded_data_train3 = decoder3.predict(encoded_data_train3)
encoded_data_valid3 = encoder3.predict(encoded_data_valid2)
decoded_data_valid3 = decoder3.predict(encoded_data_valid3)

#Save model
#Serialize model to JSON
aec3_json = aec3.to_json()
with open("data/aec3.json", "w") as json_file:
    json_file.write(aec3_json)
# serialize weights to HDF5
aec3.save_weights("data/aec3.h5")
print("Saved model to disk")

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
			
			cc1.fit(cc1_train_valid_data.input_train, cc1_train_valid_data.output_train,
            			    epochs=EPOCHS,
			                batch_size=BATCH_SIZE,
			                shuffle=True,
			                validation_data=(cc1_train_valid_data.input_valid, cc1_train_valid_data.output_valid))

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
			cc2.fit(encoded_data_train_cc1, encoded_data_train_cc1,
            			    epochs=EPOCHS,
			                batch_size=BATCH_SIZE,
			                shuffle=True,
			                validation_data=(encoded_data_valid_cc1, encoded_data_valid_cc1))

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
			
			cc3.fit(encoded_data_train_cc2, encoded_data_train_cc2,
            			    epochs=EPOCHS,
			                batch_size=BATCH_SIZE,
			                shuffle=True,
			                validation_data=(encoded_data_valid_cc2, encoded_data_valid_cc2))

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
			#stack features.....................................
			#auto_cc_features_train = np.append(auto_cc_features_train, decoded_data_train_cc, axis = 1)
			#auto_cc_features_valid = np.append(auto_cc_features_valid, decoded_data_valid_cc, axis = 1)
			#auto_cc_features_test = np.append(auto_cc_features_test, decoded_data_test_cc, axis = 1)

cc_end = time.time() 
