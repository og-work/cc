from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Reshape, Dropout
from keras.models import Model, load_model
from keras.optimizers import adam, rmsprop
from keras.applications.vgg19 import preprocess_input
import os
from numpy import genfromtxt
from matplotlib import pyplot as plt
#from utils import get_image_names_and_labels, get_parse_batch
from random import randint
from keras import backend as Keras
import numpy as np
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import model_from_json
from keras.applications.vgg19 import preprocess_input
import pdb
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
import scipy.io


if 1:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))

'''
========================================================================================================================
CONSTANTS
========================================================================================================================
'''
batch_size = 16
num_iter = 10000000
decay_step = 10000000
save_step = 1000
disp_step = 10
eval_step = 10
img_width = 224
img_height = 224
nb_train_samples = 3680
nb_validation_samples = 3669
NUMBER_OF_CLASSES = 37
EPOCHS = 100
model_path = '/nfs4/omkar/Documents/study/phd-research/codes/tf-codes/data-oxfordpets/'
train_data_dir = '/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/finetune-train-test/train/'
validation_data_dir = '/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/finetune-train-test/test/'
dataset_features_dir = '/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/'

def preprocess_input3(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = Keras.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
	
    #print data_format
	
    if data_format == 'channels_first':
	# 'RGB'->'BGR'
	x = x[::-1, :, :]
	# Zero-center by mean pixel
	x[0, :, :] -= 103.939
	x[1, :, :] -= 116.779
	x[2, :, :] -= 123.68
    else:
	# 'RGB'->'BGR'
	x = x[:, :, ::-1]
	# Zero-center by mean pixel
	x[:, :, 0] -= 103.939
	x[:, :, 1] -= 116.779
	x[:, :, 2] -= 123.68
	
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

class dataset_info:
	dataset_train_labels = np.array([])
	dataset_test_labels = np.array([])
	dataset_features = np.array([])
	def function(self):
		print "This is a class for dataset features and labels"

'''
Function to get the features from the saved model
------------------------------------------------------
-----------------------------------------------------
'''
def function_extract_features(train_data_dir, validation_data_dir):

	#load json and create model
	json_file = open('vgg19_oxfordpets.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights("vgg19_oxfordpets.h5")
	print("Loaded model from disk")

	#evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	#score = loaded_model.evaluate(X, Y, verbose=0)
	#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))  

	print loaded_model.summary()
	model_for_feature_extraction = Model(input=loaded_model.input, output=loaded_model.get_layer('my_fc2_4096').output)
	print model_for_feature_extraction.summary()

	#Sample feature extraction
	train_labels = []
	train_features = []
	label = 0
	print "Extracting features for *train* classes from finetuned VGG19"
	for root, dirs, files in os.walk(train_data_dir):
		for dirname in sorted(dirs):
			print dirname
			label = label + 1
			for file in os.listdir(os.path.join(root, dirname)):
				if file.endswith(".jpg"):
					img_path = os.path.join(os.path.join(root, dirname), file)
					#print img_path, label
					img = image.load_img(img_path, target_size=(img_width, img_height))
					x = image.img_to_array(img)
					x = np.expand_dims(x, axis=0)
					x = preprocess_input3(x)
					feature = model_for_feature_extraction.predict(x)
					feature = np.reshape(feature, 4096, )
					train_labels.append(label)
					train_features.append(feature)
	
	test_labels = []
	test_features = []
	label = 0
	print "Extracting features for *test* classes from finetuned VGG19"
	for root, dirs, files in os.walk(validation_data_dir):
		for dirname in sorted(dirs):
			print dirname
			label = label + 1
			for file in os.listdir(os.path.join(root, dirname)):
				if file.endswith(".jpg"):
					img_path = os.path.join(os.path.join(root, dirname), file)
					#print img_path, label
					img = image.load_img(img_path, target_size=(img_width, img_height))
					x = image.img_to_array(img)
					x = np.expand_dims(x, axis=0)
					x = preprocess_input3(x)
					feature = model_for_feature_extraction.predict(x)
					feature = np.reshape(feature, 4096, )
					test_labels.append(label)
					test_features.append(feature)
	
	obj_dataset_info = dataset_info()
	test_labels = np.asarray(test_labels)
	train_labels = np.asarray(train_labels)
	dummy_labels_end = [-1] * test_labels.shape[0]
	dummy_labels_start = [-1] * train_labels.shape[0]
	obj_dataset_info.dataset_train_labels = np.hstack((train_labels, dummy_labels_end))
	obj_dataset_info.dataset_test_labels = np.hstack((dummy_labels_start, test_labels))
	test_features = np.asarray(test_features)
	train_features = np.asarray(train_features)
	obj_dataset_info.dataset_features = np.vstack((train_features, test_features))
	print (obj_dataset_info.dataset_train_labels).shape	
	print (obj_dataset_info.dataset_test_labels).shape	
	print (obj_dataset_info.dataset_features).shape	
	print "Number of training samples %d"%(train_labels.shape[0])
	print "Number of testing samples %d"%(test_labels.shape[0])
	print "Total number of samples %d"%(test_labels.shape[0] + train_labels.shape[0])
	return obj_dataset_info
	

#Get the features
filename =  dataset_features_dir + 'oxfordpets_dataset_features1'
if not os.path.isfile(filename + '.mat'):
	obj_dataset_info = dataset_info()
	obj_dataset_info = function_extract_features(train_data_dir, validation_data_dir)
	scipy.io.savemat(filename, dict(dataset_features  = obj_dataset_info.dataset_features,
					dataset_train_labels = obj_dataset_info.dataset_train_labels,
					dataset_test_labels = obj_dataset_info.dataset_test_labels
					))
	print filename


'''
========================================================================================================================
MODEL DEFINITION
========================================================================================================================
'''
resnet = applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
print resnet.summary()
input_layer = Input(shape=(img_width, img_height, 3))
resnet_features = resnet(input_layer)
resnet_features = Reshape(target_shape=(25088, ), name='my_flatten')(resnet_features)
# resnet_features = Dropout(0.2)(resnet_features)
resnet_dense = Dense(4096, activation='relu', name='my_fc1_4096')(resnet_features)
resnet_dense = Dense(4096, activation='relu', name='my_fc2_4096')(resnet_dense)
resnet_prob = Dense(NUMBER_OF_CLASSES, activation='softmax', name='my_classifier')(resnet_dense)
oxfordpets_vgg19 = Model(inputs=input_layer, outputs=resnet_prob)
for layer in oxfordpets_vgg19.layers[:2]:
	layer.trainable = False
	#keras.backend.set_value(oxfordpets_vgg19.optimizer.lr, 0.0001)
for layer in oxfordpets_vgg19.layers[2:]:
	layer.trainable = True
	#keras.backend.set_value(oxfordpets_vgg19.optimizer.lr, 0.00001)
for layer in oxfordpets_vgg19.layers:
    print layer.name, layer.trainable

optimizer = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
oxfordpets_vgg19.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(oxfordpets_vgg19.summary())

# oxfordpets_vgg19 = load_model(model_path)
'''
========================================================================================================================
TRAINING
========================================================================================================================
'''
train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   preprocessing_function=preprocess_input3
                                   )

# train_datagen = ImageDataGenerator(horizontal_flip=False,
#                                    preprocessing_function=preprocess_input3
#                                    )

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input3)

train_generator = train_datagen.flow_from_directory(train_data_dir, \
                                                    target_size=(img_width, img_height), \
                                                    batch_size=batch_size, \
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, \
                                                        target_size=(img_width, img_height), \
                                                        batch_size=batch_size, \
                                                        class_mode='categorical')

history = oxfordpets_vgg19.fit_generator(generator=train_generator, \
					 steps_per_epoch=int(nb_train_samples/batch_size), \
					 epochs=EPOCHS, \
					 validation_data=validation_generator, \
					 validation_steps=int(nb_validation_samples/batch_size),\
					 initial_epoch=0)
#temp = oxfordpets_vgg19.evaluate_generator(validation_generator, int(nb_validation_samples/batch_size))

#Save model
# serialize model to JSON
model_json = oxfordpets_vgg19.to_json()
with open("vgg19_oxfordpets.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
oxfordpets_vgg19.save_weights("vgg19_oxfordpets.h5")
print("Saved model to disk") 


