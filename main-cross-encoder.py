from keras.layers import Input, Dense
from keras.models import Model
import scipy.io
import matplotlib
from keras.datasets import mnist
import numpy as np
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.callbacks import TensorBoard
print "******************************************************************************************************************************************************"
print "******************************************************************************************************************************************************"
SAMPLE_DATA = 0
EPOCHS = 500
BATCH_SIZE = 256
USE_DESKTOP = 0;
TRAIN_VALIDATION_SPLIT = 0.7

if USE_DESKTOP:
	BASE_PATH = '/nfs4/omkar/Documents/'
else:
	BASE_PATH = '/media/omkar/windows-D-drive/'


if 1 == SAMPLE_DATA:
	path_CNN_features = BASE_PATH + "/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/cnn_feat_imagenet-vgg-verydeep-19.mat"
	path_attributes = BASE_PATH + "/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/class_attributes.mat"
	features = scipy.io.loadmat(path_CNN_features)
	attributes_data = scipy.io.loadmat(path_attributes)
	attributes = attributes_data['class_attributes']
	dataset_labels = attributes_data['labels']
	visual_features_dataset = features['cnn_feat']
	visual_features_dataset = visual_features_dataset.transpose()
	train_class_labels = np.arange(1, 21, 1)
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
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# this is our input placeholder
input_img = Input(shape=(dimension_visual_data,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(dimension_visual_data, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
aec_start = time.time()
autoencoder.fit(train_samples, train_samples,
                epochs=EPOCHS,
                batch_size=256,
                shuffle=True,
                validation_data=(valid_samples, valid_samples),
				callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
aec_end = time.time()
print "aec training time: %f"%((aec_end - aec_start)/60) 
# encode and decode some digits
encoded_data_test = encoder.predict(test_samples)
decoded_data_test = decoder.predict(encoded_data_test)
encoded_data_train = encoder.predict(train_samples)
decoded_data_train = decoder.predict(encoded_data_train)
encoded_data_valid = encoder.predict(valid_samples)
decoded_data_valid = decoder.predict(encoded_data_valid)

#Save model....................................
#Serialize model to JSON
autoencoder_json = autoencoder.to_json()
with open("autoencoder.json", "w") as json_file:
    json_file.write(autoencoder_json)
# serialize weights to HDF5
autoencoder.save_weights("autoencoder.h5")
print("Saved model to disk")

#..............CC...............................
#Get data for CC ................................
number_of_cc = number_of_train_classes * number_of_train_classes - number_of_train_classes
encoding_dimension_CC = 32
input_CC = Input(shape=(dimension_visual_data,))
encoded_CC = Dense(encoding_dimension_CC, activation='relu')(input_CC)
decoded_CC = Dense(dimension_visual_data, activation='sigmoid')(encoded_CC)
cross_coder = Model(input_CC, decoded_CC)
encoder_CC = Model(input_CC, encoded_CC)
encoded_input_CC = Input(shape=(encoding_dimension_CC,))
decoder_layer_CC = cross_coder.layers[-1]
decoder_CC = Model(encoded_input_CC, decoder_layer_CC(encoded_input_CC))
cross_coder.compile(optimizer='adadelta', loss='binary_crossentropy')

cross_coders_train_data_input = []
cross_coders_train_data_output = []
auto_cc_features_train = decoded_data_train
auto_cc_features_test = decoded_data_test
auto_cc_features_valid = decoded_data_valid

cc_start = time.time() 
for classI in train_class_labels:
	for classJ in train_class_labels:
		input_sample_indices_train = np.array([])
		output_sample_indices_train = np.array([])
		input_sample_indices_valid = np.array([])
		output_sample_indices_valid = np.array([])
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
			for classI_sample in indices_classI_samples_train:
				classI_sample_array = np.empty(indices_classJ_samples_train.size)
				classI_sample_array.fill(classI_sample)
				input_sample_indices_train = np.concatenate((input_sample_indices_train, classI_sample_array), axis = 0)
				output_sample_indices_train = np.concatenate((output_sample_indices_train, indices_classJ_samples_train), axis = 0)

			#Prepare validation data for CC
			for classI_sample in indices_classI_samples_valid:
				classI_sample_array = np.empty(indices_classJ_samples_valid.size)
				classI_sample_array.fill(classI_sample)
				input_sample_indices_valid = np.concatenate((input_sample_indices_valid, classI_sample_array), axis = 0)
				output_sample_indices_valid = np.concatenate((output_sample_indices_valid, indices_classJ_samples_valid), axis = 0)
			#print(input_sample_indices_train)							
			#print(output_sample_indices_train)
			#crossCodersTrainDataInput.append(input_sample_indices_train)							
			#crossCodersTrainDataOutput.append(output_sample_indices_train)							
			input_samples_for_CC_train = visual_features_dataset[input_sample_indices_train.astype(int), :]
			output_samples_for_CC_train = visual_features_dataset[output_sample_indices_train.astype(int), :]
			input_samples_for_CC_valid = visual_features_dataset[input_sample_indices_valid.astype(int), :]
			output_samples_for_CC_valid = visual_features_dataset[output_sample_indices_valid.astype(int), :]
			#print('Input samples for cross coder')
			#print(inputSamplesForCCTrain)	
			#print('Output samples for cross coder')
			#print(outputSamplesForCCTrain)	
			#print('Input samples for cross coder valid')
			#print(inputSamplesForCCValid)	
			#print('Output samples for cross coder valid')
			#print(outputSamplesForCCValid)	
			cross_coder.fit(input_samples_for_CC_train, output_samples_for_CC_train,
            			    epochs=EPOCHS,
			                batch_size=BATCH_SIZE,
			                shuffle=True,
			                validation_data=(input_samples_for_CC_valid, output_samples_for_CC_valid))
			#Save models to JSON
			modelName = 'CC_' + str(classI) + '_' + str(classJ) + '.json' 
			crossCoder_json = cross_coder.to_json()
			with open(modelName, "w") as json_file:
			    json_file.write(crossCoder_json)
			# serialize weights to HDF5
			modelNameh5 = 'CC_' + str(classI) + '_' + str(classJ) + '.h5'
			cross_coder.save_weights(modelNameh5)
			
			#Get CC features for training samples
			decoded_data_train_CC = cross_coder.predict(train_samples)	
			decoded_data_valid_CC = cross_coder.predict(valid_samples)	
			decoded_data_test_CC = cross_coder.predict(test_samples)	

			#stack features.....................................
			auto_cc_features_train = np.append(auto_cc_features_train, decoded_data_train_CC, axis = 1)
			auto_cc_features_valid = np.append(auto_cc_features_valid, decoded_data_valid_CC, axis = 1)
			auto_cc_features_test = np.append(auto_cc_features_test, decoded_data_test_CC, axis = 1)

cc_end = time.time() 
print "cc train time: %f" %((cc_end - cc_start)/60)
# use Matplotlib (don't ask)
if 0:
	n = 10  # how many digits we will display
	plt.figure(figsize=(20, 4))
	for i in range(n):
    	# display original
	    ax = plt.subplot(2, n, i + 1)
    	#plt.imshow(x_test[i].reshape(28, 28))
	    plt.imshow(trainSamples[i, :].reshape(64, 64))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	
	    # display reconstruction
	    ax = plt.subplot(2, n, i + 1 + n)
	    plt.imshow(testSamples[i, :].reshape(64, 64))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()
