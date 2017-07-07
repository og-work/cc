from keras.layers import Input, Dense
from keras.models import Model
import scipy.io
import matplotlib
from keras.datasets import mnist
import numpy as np
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

print "******************************************************************************************************************************************************"
print "******************************************************************************************************************************************************"
SAMPLE_DATA = 1
EPOCHS = 1
BATCH_SIZE = 256

if 0 == SAMPLE_DATA:
	pathCNNFeatures = "/nfs4/omkar/Documents/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/cnn_feat_imagenet-vgg-verydeep-19.mat"
	pathAttributes = "/nfs4/omkar/Documents/study/phd-research/data/code-data/semantic-similarity/cnn-features/aPY/class_attributes.mat"
	features = scipy.io.loadmat(pathCNNFeatures)
	attributesData = scipy.io.loadmat(pathAttributes)
	attributes = attributesData['class_attributes']
	datasetLabels = attributesData['labels']
	visualFeaturesDataset = features['cnn_feat']
	visualFeaturesDataset = visualFeaturesDataset.transpose()
	trainClassLabels = np.arange(1, 21, 1)
	testClassLabels = np.arange(21, 33, 1)
else:
	datasetLabels = np.array([1, 1, 1, 1, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3])
	attributes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
	visualFeaturesDataset = np.array([[11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33],
									  [11, 14, 15, 16, 17, 21, 12, 24, 25, 31, 34, 36, 37, 38, 39, 32, 22, 33]])
	visualFeaturesDataset = visualFeaturesDataset.transpose()
	trainClassLabels = np.arange(1, 4, 1)
	testClassLabels = np.arange(2, 3, 1)
	numberOfTrainClasses = np.size(trainClassLabels)
	print(trainClassLabels)
	print(testClassLabels)

numberOfTrainClasses = np.size(trainClassLabels)
numberOfTestClasses = np.size(testClassLabels)
dimensionVisualData = visualFeaturesDataset.shape[1]
numberOfTrainingSamples = visualFeaturesDataset.shape[0]
dimensionAttributes = attributes.shape[1]
numberOfClasses = attributes.shape[0]
print "Dataset visual features shape is: %d X %d" % visualFeaturesDataset.shape
print "Dimension of visual data: %d" %dimensionVisualData
print "Number of training samples: %d" %numberOfTrainingSamples
print "Dimension of attributes: %d" %dimensionAttributes
print "Number of classes: %d" %numberOfClasses

#Get training samples
trainSampleIndices = np.array([])
for classIndex in trainClassLabels: 
	indices = np.flatnonzero(datasetLabels == classIndex)
	trainSampleIndices = np.concatenate((trainSampleIndices, indices), axis = 0)
trainSamples = visualFeaturesDataset[trainSampleIndices.astype(int), :]

#Get testing samples
testSampleIndices = np.array([])
for classIndex in testClassLabels: 
	indices = np.flatnonzero(datasetLabels == classIndex)
	#add new feature as a new column
	testSampleIndices = np.concatenate((testSampleIndices, indices), axis = 0)
testSamples = visualFeaturesDataset[testSampleIndices.astype(int), :]

print "Train features shape: %d X %d" %trainSamples.shape
print "Test features shape: %d X %d" %testSamples.shape

if SAMPLE_DATA:
	for k in [0, 1, 2, 3]:
		print(trainSamples[:, k])
	for k in [0, 1]:
		print(testSamples[:, k])
	print(trainSampleIndices.shape)
	print(type(trainSampleIndices))
	print "Dataset labels shape is: %d" %datasetLabels.shape
	print "Dataset labels type is: %s " % type(datasetLabels)
	print "Training features: %d X %d" %trainSamples.shape
	print "Testing features: %d X %d" %testSamples.shape
	print(trainSampleIndices)
	print(testSampleIndices)

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(dimensionVisualData,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(dimensionVisualData, activation='sigmoid')(encoded)

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
autoencoder.fit(trainSamples, trainSamples,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(testSamples, testSamples))

# encode and decode some digits
# note that we take them from the *test* set
encodedData = encoder.predict(testSamples)
decodedData = decoder.predict(encodedData)
print "Decoded data shape: %d %d" %decodedData.shape

#Serialize model to JSON
autoencoder_json = autoencoder.to_json()
with open("autoencoder.json", "w") as json_file:
    json_file.write(autoencoder_json)
# serialize weights to HDF5
autoencoder.save_weights("autoencoder.h5")
print("Saved model to disk")

#..............Cross-coders.....................
#Get data for cross coders......................
encodingDimensionCC = 32
inputCC = Input(shape=(dimensionVisualData,))
encodedCC = Dense(encodingDimensionCC, activation='relu')(inputCC)
decodedCC = Dense(dimensionVisualData, activation='sigmoid')(encodedCC)
crossCoder = Model(inputCC, decodedCC)
encoder = Model(inputCC, encodedCC)
encodedInputCC = Input(shape=(encodingDimensionCC,))
decoderLayerCC = crossCoder.layers[-1]
decoderCC = Model(encodedInputCC, decoderLayerCC(encodedInputCC))
crossCoder.compile(optimizer='adadelta', loss='binary_crossentropy')


crossCodersTrainDataInput = []
crossCodersTrainDataOutput = []
 
for classI in trainClassLabels:
	for classJ in trainClassLabels:
		inputSampleIndicesTrain = np.array([])
		outputSampleIndicesTrain = np.array([])
		inputSampleIndicesValid = np.array([])
		outputSampleIndicesValid = np.array([])
		print "**************************************"
		if classI!=classJ:
			indicesClassISamples = np.flatnonzero(datasetLabels == classI)
			indicesClassJSamples = np.flatnonzero(datasetLabels == classJ)
			numberOfSamplesClassIForTrain = int(0.7 * np.size(indicesClassISamples))
			numberOfSamplesClassJForTrain = int(0.7 * np.size(indicesClassJSamples))
			indicesClassISamplesTrain = indicesClassISamples[:numberOfSamplesClassIForTrain]
			indicesClassISamplesValid = indicesClassISamples[(numberOfSamplesClassIForTrain):]
			indicesClassJSamplesTrain = indicesClassJSamples[:numberOfSamplesClassJForTrain]
			indicesClassJSamplesValid = indicesClassJSamples[(numberOfSamplesClassJForTrain):]
			print "classI %d classJ %d indices %d %d %d %d" %(classI, classJ, indicesClassISamples.size, indicesClassJSamples.size, \
				numberOfSamplesClassIForTrain, numberOfSamplesClassJForTrain)
			print(indicesClassISamples)
			print(indicesClassISamplesTrain)
			print(indicesClassISamplesValid)
			print(indicesClassJSamples)
			print(indicesClassJSamplesTrain)
			print(indicesClassJSamplesValid)
			#Prepare train data for cross-coder
			for classISample in indicesClassISamplesTrain:
				classISamples = np.empty(indicesClassJSamplesTrain.size)
				classISamples.fill(classISample)
				inputSampleIndicesTrain = np.concatenate((inputSampleIndicesTrain, classISamples), axis = 0)
				outputSampleIndicesTrain = np.concatenate((outputSampleIndicesTrain, indicesClassJSamplesTrain), axis = 0)
			#Prepare validation data for cross-coder
			for classISample in indicesClassISamplesValid:
				classISamples = np.empty(indicesClassJSamplesValid.size)
				classISamples.fill(classISample)
				inputSampleIndicesValid = np.concatenate((inputSampleIndicesValid, classISamples), axis = 0)
				outputSampleIndicesValid = np.concatenate((outputSampleIndicesValid, indicesClassJSamplesValid), axis = 0)
			print(inputSampleIndicesTrain)							
			print(outputSampleIndicesTrain)
			#crossCodersTrainDataInput.append(inputSampleIndicesTrain)							
			#crossCodersTrainDataOutput.append(outputSampleIndicesTrain)							
			inputSamplesForCCTrain = visualFeaturesDataset[inputSampleIndicesTrain.astype(int), :]
			outputSamplesForCCTrain = visualFeaturesDataset[outputSampleIndicesTrain.astype(int), :]
			inputSamplesForCCValid = visualFeaturesDataset[inputSampleIndicesValid.astype(int), :]
			outputSamplesForCCValid = visualFeaturesDataset[outputSampleIndicesValid.astype(int), :]
			print('Input samples for cross coder')
			print(inputSamplesForCCTrain)	
			print('Output samples for cross coder')
			print(outputSamplesForCCTrain)	
			print('Input samples for cross coder valid')
			print(inputSamplesForCCValid)	
			print('Output samples for cross coder valid')
			print(outputSamplesForCCValid)	
			crossCoder.fit(inputSamplesForCCTrain, outputSamplesForCCTrain,
            			    epochs=EPOCHS,
			                batch_size=BATCH_SIZE,
			                shuffle=True,
			                validation_data=(inputSamplesForCCValid, outputSamplesForCCValid))
			#Save models to JSON
			modelName = 'crossCoder_' + str(classI) + '_' + str(classJ) + '.json' 
			crossCoder_json = crossCoder.to_json()
			with open(modelName, "w") as json_file:
			    json_file.write(crossCoder_json)
			# serialize weights to HDF5
			modelNameh5 = 'crossCoder_' + str(classI) + '_' + str(classJ) + '.h5'
			crossCoder.save_weights(modelNameh5)
			print("Saved model to disk")
			
print(crossCodersTrainDataInput)
print(crossCodersTrainDataOutput)
print "accessing list ..."
#print(crossCodersTrainDataInput[2][2])

#Train cross-coders...........................
 

trainSamplesInputCross = visualFeaturesDataset[inputSampleIndicesTrain.astype(int), :]
trainSamplesOutputCross = visualFeaturesDataset[outputSampleIndicesTrain.astype(int), :]
# use Matplotlib (don't ask)

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
