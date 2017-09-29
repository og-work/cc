
#.................................oxfordpets_conv_aec_main.py................................


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import scipy.io
import pdb
import matplotlib.pyplot as plt
from keras.optimizers import adam, rmsprop, SGD, Adam
from keras import optimizers
import itertools
import os

if 1:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))

img_width = 224
img_height = 224
num_channels = 3
NUM_CLASSES = 5
TR_TS_VA_SPLIT = np.array([0.6, 0.2, 0.2])
EPOCHS = 200
BATCH_SIZE = 8
OPTIMIZER_TYPE = 'SGD'
LEARNING_RATE = 0.5
BATCH_SIZE = 16
DECAY = 1e-6
MOMENTUM =0.09

#loss = 'binary_crossentropy'
loss = 'mse'


PATH_DATA = "/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/oxfordpets_dataset_conv_aec_features"

def function_load_data(data_path):
	print "Loading features from %s" %data_path
	tmp = scipy.io.loadmat(data_path)
        visual_features_dataset = tmp['dataset_features']
	dataset_train_labels = tmp['dataset_train_labels']
	dataset_test_labels = tmp['dataset_test_labels']
	
	return (visual_features_dataset, dataset_train_labels, dataset_test_labels)

def function_get_cc_data(visual_dataset_features, dataset_train_labels, dataset_test_labels, classI, classJ):

	indices_classI_samples = np.flatnonzero(dataset_train_labels == classI)
	indices_classJ_samples = np.flatnonzero(dataset_train_labels == classJ)
	indices_classI_samples_test = np.flatnonzero(dataset_test_labels == classI)

	min_num_samples_classI_classJ  = min(np.size(indices_classI_samples), np.size(indices_classJ_samples))
	num_train_samples = int(TR_TS_VA_SPLIT[0] * min_num_samples_classI_classJ)	
	num_valid_samples = int(TR_TS_VA_SPLIT[1] * min_num_samples_classI_classJ)	
	num_test_samples = int(TR_TS_VA_SPLIT[2] * np.size(indices_classJ_samples))	
	
	indices_classI_train = indices_classI_samples[:num_train_samples]
	indices_classI_valid = indices_classI_samples[num_train_samples:num_train_samples + num_valid_samples]
	indices_classJ_train = indices_classJ_samples[:num_train_samples]
	indices_classJ_valid = indices_classJ_samples[num_train_samples:num_train_samples + num_valid_samples]

	if classI != classJ:
		indices_classI_train_perm = np.repeat(indices_classI_train, np.size(indices_classI_train))
		indices_classJ_train_perm = np.tile(indices_classJ_train, (np.size(indices_classJ_train), ))
	else:
		indices_classI_train_perm = indices_classI_train
		indices_classJ_train_perm = indices_classJ_train
	
	train_input = visual_dataset_features[indices_classI_train_perm, :]	
	train_output = visual_dataset_features[indices_classJ_train_perm, :]	
	valid_input = visual_dataset_features[indices_classI_valid, :]	
	valid_output = visual_dataset_features[indices_classJ_valid, :]	
	test_input = visual_dataset_features[indices_classI_samples_test, :]	

	return (train_input, train_output, valid_input, valid_output, test_input)

input_img = Input(shape=(img_width, img_height, 3))  # adapt this if using `channels_first` image data format
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='my_l1')(input_img)
x = MaxPooling2D((2, 2), padding='same', name='my_l2')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='my_l3')(x)
x = MaxPooling2D((2, 2), padding='same', name='my_l4')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='my_l5')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='my_l6')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same', name='my_l7')(encoded)
x = UpSampling2D((2, 2), name='my_l8')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='my_l9')(x)
x = UpSampling2D((2, 2), name='my_l10')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='my_l11')(x)
x = UpSampling2D((2, 2), name='my_l12')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='my_l13')(x)

autoencoder = Model(input_img, decoded)
if OPTIMIZER_TYPE == 'SGD':
        print "Using SGD optimizer..."
        OPTIMIZER = SGD(lr=LEARNING_RATE, decay=DECAY, momentum = MOMENTUM, nesterov=True)
elif OPTIMIZER_TYPE == 'adam':
        print "Using adam optimizer..."
        OPTIMIZER = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

autoencoder.compile(optimizer=OPTIMIZER, loss='mse')#, metrics=['mse'])


(visual_dataset_features, dataset_train_labels, dataset_test_labels) = function_load_data(PATH_DATA)
(train_input, train_output, valid_input, valid_output, test_input) = function_get_cc_data(visual_dataset_features, dataset_train_labels, dataset_test_labels, 1, 1)

print autoencoder.summary()
print "Data for training/validation with size:"
print train_input.shape
print train_output.shape
print valid_input.shape
print valid_output.shape

#pdb.set_trace()
#train_input = np.reshape(x_train, (len(x_train), img_width, img_height, 3))  # adapt this if using `channels_first` image data format
#train_output = np.reshape(x_test, (len(x_test), img_width, img_height, 3))  # adapt this if using `channels_first` image data format

autoencoder.fit(train_input, train_output,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(valid_input, valid_output),
             )

decoded_imgs = autoencoder.predict(test_input)
print test_input.shape
print decoded_imgs.shape
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_input[i, :, :, :])#.reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i, :, :, :])#.reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
