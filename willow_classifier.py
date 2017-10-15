
# coding: utf-8

# In[1]:


#...........................willow_classifier.py................................

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
#import pandas

import random
from numpy import linalg as LA
import scipy.io
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pdb
import random

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
# In[5]:

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

if 1:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))


# In[10]:

'''
---------------------------------------------------
'''
dataset_name = 'WILLOW'
dim_feature = 100
#path_cross_features = '../data-willow/data2_hemachandra/'
path_cross_features = '/home/SharedData/omkar/study/phd-research/codes/tf-codes/data-willow/data3_hemachandra/'
'''
---------------------------------------------------
'''

LIST_OF_TRAIN_OPTIONS =  ['USE_SAVED_WEIGHTS', 'USE_SAVED_WEIGHTS_AND_RETRAIN', 'TRAIN']
TRAIN_OPTION = LIST_OF_TRAIN_OPTIONS[1]
PERCENTAGE_DROP_OUT1 = 0.5
PERCENTAGE_DROP_OUT2 = 0.5
NUMBER_OF_CLASSES = 7
list_methods = ['BASELINE', 'FULL', 'CROSS']
method = str(list_methods[2])
if method == 'BASELINE':
	DIM_INPUT = dim_feature
elif method == 'FULL':
	DIM_INPUT = NUMBER_OF_CLASSES * dim_feature
elif method == 'CROSS':
	DIM_INPUT = (NUMBER_OF_CLASSES - 1) * dim_feature
DIM_FC1 = int(0.5 * DIM_INPUT)
DIM_FC2 = int(0.5 * DIM_FC1)
DIM_FC3 = int(0.5 * DIM_FC2)
EPOCHS = 5
LEARNING_RATE = 0.0001 
MIN_LR_RATE = 0.01
OPTIMIZER_TYPE = 'adam'#SGD/adam
BATCH_SIZE = 32
DECAY = 1e-7 
MOMENTUM =0.09

SUBDATA_SIZE_TRAIN = 490
SUBDATA_SIZE_TEST = 633

'''
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
'''
def function_is_train(val_history):
	#pdb.set_trace()
	val_history = np.asarray(val_history)
	pateince = 3
	tolerance = 0.01
	train = 0
	n_epochs = val_history.shape[0]
	latest_val_acc = val_history[n_epochs - 1]
	if n_epochs > 5:
		for k in range(pateince):
			if (latest_val_acc > tolerance +  val_history[n_epochs - 2 - k]) and (latest_val_acc < tolerance +  val_history[n_epochs - 2 - k]):
				train = 1
				break
	else:
		train = 1
	print train	
	return train

def get_covariance_matrix(data):
	mean = np.mean(data, axis=0)
	cov = np.zeros((data.shape[1], data.shape[1]))
	for i in range(data.shape[0]):
		cov = cov + np.matmul(np.transpose(data[i, :]), data[i, :])
	cov = cov / data.shape[0]	
	return cov        

def function_scatterMatrix_nclass(train_data, train_labels):

#withinClassScatter maytrix for fixed two sample size classes-
# optimized for fast compuation using matrix outerproduct.
	noFeatures = train_data.shape[1]
	noClass = NUMBER_OF_CLASSES
	Xk = []#cell(noClass,1) 
	nk = np.zeros((noClass,1))
	nk = np.reshape(nk, (1, nk.shape[0]))
	muk = np.zeros((noClass,noFeatures))
	m = np.zeros((noFeatures,1))
	for j in range(noClass):
		sample_indices = np.flatnonzero(train_labels == (j))
		
		X1 = train_data[sample_indices,:]
		Xk.append(X1)
		nk[0, j] = X1.shape[0]
		muk[j,:] = np.mean(X1,axis=0)
		m = m + nk[0, j]*muk[j,:]


	N = np.sum(nk);
	m = m/N;
	Sb = np.zeros((noFeatures,noFeatures))
	for j in range(noClass):
		Sb = Sb + nk[0, j]*np.transpose((muk[j,:] - m))*(muk[j,:] - m)

	Sb = Sb/N
	Sw = np.zeros((noFeatures,noFeatures))
	for j in range(noClass):
	    print "class %d "%j	
	    covariance_mat = get_covariance_matrix(Xk[j])	
	    Sw = Sw + nk[0, j]*covariance_mat

	Sw = Sw/N
	return Sw, Sb

def function_classifier_mmc(data_train, labels_train, data_test, labels_test):
	#finds optimal discriminants to reduce the feature dimension using
	#maximum margin classifier
	Sw, Sb = function_scatterMatrix_nclass(data_train, labels_train);
	eigenValues, eigenVectors = LA.eig((Sb + np.transpose(Sb))/2 - (Sw + np.transpose(Sw)) / 2) 
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	# dmsn = size(X,2);   num_train = size(X1,1)+size(X2,1);
	#             for k = 1:dmsn
	#%                 W = Vr(:,1:k);
	#%                 Y1 = X1*W; Y2 = X2*W;
	#%                 accrcy(k) = 
	#%             end
	#%             [~,best_k] = max(accrcy);
	for i in range(eigenValues.shape[0]):
		if eigenValues[i] < 0:
			best_k = i -1
			break

	best_k = np.maximum(0,best_k)
	W = eigenVectors[:,0:best_k];
	#% X3 = X(Label_sp==0,:);
	#% Y1 = X1*W;            Y2 = X2*W;            Y3 = X3*W;
	#% Y = X*W;
	return W
	
def function_use_knn_classifier(data_train, labels_train, data_test, labels_test):
	print "-------------------------------Using KNN classifier--------------------------------------"
	n_nbrs = 3
	neigh = KNeighborsClassifier(n_neighbors=n_nbrs)
	neigh.fit(data_train, labels_train) 
	predicted_labels_test = neigh.predict(data_test)
	#print(neigh.predict_proba([[0.9]]))
	accuracy = accuracy_score(predicted_labels_test, labels_test)
        print "KNN accuracy %f with %d nbrs: "%(accuracy, n_nbrs)
	return accuracy 



def function_use_svm_classifier(data_train, labels_train, data_test, labels_test):
	#X, y = make_classification(n_features=data_train.shape[1], random_state=0)
	print "-------------------------------Using SVM classifier--------------------------------------"
	full_data = np.vstack((data_train, data_test))
	#scaler = StandardScaler()
	#data_train = scaler.fit(data_train)
	#data_test = scaler.transform(data_test)
	
	labels_train = labels_train.flatten()
	labels_test = labels_test.flatten()
	
	#clf = LinearSVC(random_state=1)
	#print clf.fit(data_train, labels_train)
	#predicted_labels_test = clf.predict(data_test)
	#accuracy = accuracy_score(predicted_labels_test, labels_test)
	#print "SVM accuracy: %f"%(accuracy)
	
	best_C = 2
	best_gamma = 0.001
	#------------------Grid search for para-----------------------------------------------------------
	if 1:
		C_range = np.logspace(-2, 10, 13)#base=10.0)
		gamma_range = np.logspace(-9, 3, 13)#base=10.0)
		param_grid = dict(gamma=gamma_range, C=C_range)
		k = 3
		print "Using %d fold validation"%k
		start_cvl = time.time()
		cv = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
		grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, verbose=2)
		grid.fit(data_train, labels_train)
		print("The best parameters are %s with a score of %0.2f"
		      % (grid.best_params_, grid.best_score_))
		end_cvl = time.time()
		print "Cross-validation/hyperparameter tuning time %f"%((end_cvl - start_cvl)/60)
		best_C = grid.best_params_['C']
		best_gamma = grid.best_params_['gamma']
	#------------------Grid search for para------------------------------------------------------------
	
	
	clf = SVC(C=best_C, gamma=best_gamma, probability=True)
        print clf.fit(data_train, labels_train)
	predicted_labels_test = clf.predict(data_test)
	#predicted_scores_test = clf.predict_proba(data_test)
	predicted_scores = clf.decision_function(data_test)
	accuracy = accuracy_score(predicted_labels_test, labels_test)
	average_precision = average_precision_score(labels_test, predicted_scores)
        #labels_test_cate = label_binarize(labels_test, classes=[0,1, 2])
	#roc_score = roc_auc_score(labels_test, predicted_scores)
	print "WILLOW: SVM accuracy: %f Avg Precision %f "%(accuracy, average_precision)

	return average_precision

def function_use_svm_binary_classifier(data_train, labels_train, data_test, labels_test, this_class_label):
	labels_train = labels_train + 1
	labels_test = labels_test + 1
	this_class_label = this_class_label + 1
	labels_bin_train = labels_train
	labels_bin_train[labels_bin_train != this_class_label] = 0
	labels_bin_train[labels_bin_train == this_class_label] = 1
	labels_bin_test = labels_test
	labels_bin_test[labels_bin_test != this_class_label] = 0
	labels_bin_test[labels_bin_test == this_class_label] = 1
	avg_prec = function_use_svm_classifier(data_train, labels_bin_train, data_test, labels_bin_test)
	print this_class_label, avg_prec
	
	return avg_prec
# In[7]:


dim_str = str(dim_feature) + '_500_' 
filename = path_cross_features + 'willow_' + dim_str + 'aec_features_all_classes_tr_' + str(NUMBER_OF_CLASSES)
tmp = scipy.io.loadmat(filename)
data_train_aec = tmp['cross_feautures_tr']
print 'NUmber of training samples %d'%data_train_aec.shape[0]

filename = path_cross_features + 'willow_' + dim_str + 'aec_features_all_classes_vl_' + str(NUMBER_OF_CLASSES)
tmp = scipy.io.loadmat(filename)
data_valid_aec = tmp['cross_feautures_val']
print 'NUmber of valid samples %d'%data_valid_aec.shape[0]

filename = path_cross_features + 'willow_' + dim_str + 'aec_features_all_classes_ts_' + str(NUMBER_OF_CLASSES)
tmp = scipy.io.loadmat(filename)
data_test_aec = tmp['cross_feautures_ts']
print 'NUmber of testing samples %d'%data_test_aec.shape[0]

data_train_valid_aec = np.vstack((data_train_aec, data_valid_aec))

path_cross_features_labels = path_cross_features
filename = path_cross_features_labels + 'willow_' + dim_str + 'feat_fusion_clsfr_train_labels'
tmp = scipy.io.loadmat(filename)
train_labels_all = tmp['train_labels']
train_labels_all = train_labels_all - 1

filename = path_cross_features_labels + 'willow_' + dim_str + 'feat_fusion_clsfr_valid_labels'
tmp = scipy.io.loadmat(filename)
valid_labels_all = tmp['valid_labels']
valid_labels_all = valid_labels_all - 1

train_valid_labels_all = np.hstack((train_labels_all, valid_labels_all))

filename = path_cross_features_labels + 'willow_' + dim_str + 'feat_fusion_clsfr_test_labels'
tmp = scipy.io.loadmat(filename)
test_labels_all = tmp['test_labels']
test_labels_all = test_labels_all - 1

print "Number of test samples are %d %d"%test_labels_all.shape
all_classfet = [];
cnt = 0


if method == 'BASELINE':
	print "---------------------------------Method : %s ----------------------------------------------"%method
	data_train_features = data_train_aec
	data_train_valid_features = data_train_valid_aec
	data_test_features = data_test_aec
	FEATURE_DIM_PER_CLASS = dim_feature
else: 
	for cls in range(NUMBER_OF_CLASSES):
	    filename = path_cross_features + 'willow_' + dim_str + 'cec_features_class_tr_' + str(cls + 1) 
	    print "Loading %s"%filename
	    tmp = scipy.io.loadmat(filename)
	    data_train_cec = tmp['cross_feautures_tr']

	    filename = path_cross_features + 'willow_' + dim_str + 'cec_features_class_vl_' + str(cls + 1) 
	    print "Loading %s"%filename
	    tmp = scipy.io.loadmat(filename)
	    data_valid_cec = tmp['cross_feautures_val']

	    filename = path_cross_features + 'willow_' + dim_str + 'cec_features_class_ts_' + str(cls + 1) 
	    print "Loading %s"%filename
	    tmp = scipy.io.loadmat(filename)
	    data_test_cec = tmp['cross_feautures_ts']

	    st_aec = (cls) * dim_feature
	    end_aec = st_aec + dim_feature - 1

	    data_train_cec_this_class = data_train_cec
	    data_train_valid_cec_this_class = np.vstack((data_train_cec, data_valid_cec))
	    data_test_cec_this_class = data_test_cec
	    #pdb.set_trace()	

	    if cnt == 0:
		cnt = 1
		data_train_cec_all_classes = data_train_cec_this_class
		data_train_valid_cec_all_classes = data_train_valid_cec_this_class
		data_test_cec_all_classes = data_test_cec_this_class
	    else:
		data_train_cec_all_classes = np.hstack((data_train_cec_all_classes, data_train_cec_this_class));
		data_train_valid_cec_all_classes = np.hstack((data_train_valid_cec_all_classes, data_train_valid_cec_this_class));
		data_test_cec_all_classes = np.hstack((data_test_cec_all_classes, data_test_cec_this_class));

	    print "Stacking features for class %d, dimension %d"%((cls + 1), data_test_cec_all_classes.shape[1])
	if method == 'FULL':    
		data_train_features = np.hstack((data_train_cec_all_classes, data_train_aec))
		data_train_valid_features = np.hstack((data_train_valid_cec_all_classes, data_train_valid_aec))
		data_test_features = np.hstack((data_test_cec_all_classes, data_test_aec))
		FEATURE_DIM_PER_CLASS = (NUMBER_OF_CLASSES) * dim_feature
	elif method == 'CROSS':
		FEATURE_DIM_PER_CLASS = (NUMBER_OF_CLASSES - 1) * dim_feature
		data_train_features = data_train_cec_all_classes
		data_train_valid_features = data_train_valid_cec_all_classes
		data_test_features = data_test_cec_all_classes

#Make data zero mean and unit variance
scaler = StandardScaler()
data_train_valid_features_s = scaler.fit_transform(data_train_valid_features)
data_test_features_s = scaler.transform(data_test_features)
train_valid_labels_all = train_valid_labels_all.flatten()
test_labels_all = test_labels_all.flatten()
print data_train_valid_features.shape
print train_valid_labels_all.shape

'''
-------------------------------------------------------------------------------------------------
---------------------------------CLASSIFIERS-----------------------------------------------------
-------------------------------------------------------------------------------------------------
'''
avg_precision_array = np.zeros(NUMBER_OF_CLASSES)
if 0:
	for i in range(NUMBER_OF_CLASSES):
		avg_prec = function_use_svm_binary_classifier(data_train_valid_features, train_valid_labels_all, data_test_features, test_labels_all, i)
		avg_precision_array[i] = avg_prec
		print avg_precision_array[i]
print "AP:WILLOWf" 
print avg_precision_array
print "mAP : WILLOW: SVM %f"%(np.mean(avg_precision_array))
pdb.set_trace()
#accuracy_svm = function_use_svm_classifier(data_train_valid_features, train_valid_labels_all, data_test_features, test_labels_all)
#accuracy_knn = function_use_knn_classifier(data_train_valid_features, train_valid_labels_all, data_test_features, test_labels_all)
#function_classifier_mmc(data_train_valid_features, train_valid_labels_all, data_test_features, test_labels_all)
# In[11]:

print "-------------------------------Using MLP classifier--------------------------------------"

#Input layers
print "Defining input layers"
input_1 = Input(shape=(DIM_INPUT,))
input_2 = Input(shape=(DIM_INPUT,))
input_3 = Input(shape=(DIM_INPUT,))
input_4 = Input(shape=(DIM_INPUT,))
input_5 = Input(shape=(DIM_INPUT,))
input_6 = Input(shape=(DIM_INPUT,))
input_7 = Input(shape=(DIM_INPUT,))


print "Defining FC1 layers"
fc1_1 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_1)
fc1_2 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_2)
fc1_3 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_3)
fc1_4 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_4)
fc1_5 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_5)
fc1_6 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_6)
fc1_7 = Dense(DIM_FC1, kernel_initializer='normal', activation='tanh')(input_7)


#FC2layer for all classes
print "Defining FC2 layers"
concat_features = [fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, fc1_6, fc1_7]
fc2_input = keras.layers.concatenate(concat_features)
fc2_output = Dense(DIM_FC2, kernel_initializer='normal', activation='tanh')(fc2_input)
drop_out1 = Dropout(PERCENTAGE_DROP_OUT1)(fc2_output)
#----------------------
#fc3_output = Dense(DIM_FC3, kernel_initializer='normal', activation='tanh')(drop_out1)
#drop_out2 = Dropout(PERCENTAGE_DROP_OUT2)(fc3_output)
#----------------------

#Softmax layer
print "Defining output layers"
output = Dense(NUMBER_OF_CLASSES, kernel_initializer='normal', activation='softmax')(drop_out1)

#Define feature fusion model
print "Defining input"
input_array = [input_1, input_2, input_3, input_4, input_5, input_6, input_7]
model_feature_fusion = Model(inputs=input_array, outputs=output)

if OPTIMIZER_TYPE == 'SGD':	
    print "Using SGD optimizer..."
    OPTIMIZER = SGD(lr=LEARNING_RATE, decay=DECAY, momentum = MOMENTUM, nesterov=True)
elif OPTIMIZER_TYPE == 'adam':
    print "Using adam optimizer..."
    OPTIMIZER = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=DECAY)

#------------------------------------CALL BACKS-----------------------------------------------------------

#To save weights for each imporovment
#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#To save so far best weights
filepath_model= path_cross_features + dataset_name + '_' + str(dim_feature) + '_' + method +  "_weights-best.hdf5"
checkpoint = ModelCheckpoint(filepath_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=MIN_LR_RATE)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0, patience=1, verbose=1, mode='auto')
callbacks_list = [checkpoint, reduce_lr]
#----------------------------------------------------------------------------------------------------------



'''
-----------------------------------------------------------------------------------------------------------
                                                    TRAINING
-----------------------------------------------------------------------------------------------------------
                                                   
'''
train = 1
mean_val_acc_full_data_history = []
if TRAIN_OPTION == 'USE_SAVED_WEIGHTS':
        print "Using saved weights from %s"%filepath_model
	model_feature_fusion.load_weights(filepath_model)
	model_feature_fusion.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
else:
	if TRAIN_OPTION == 'USE_SAVED_WEIGHTS_AND_RETRAIN':
		model_feature_fusion.load_weights(filepath_model)
		print "Loading weights from %s"%filepath_model
                print "Using saved weights and re-training"
	else:
		print "Training from scratch"

	model_feature_fusion.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
	
	n_subdata = np.ceil(train_valid_labels_all.shape[0] / SUBDATA_SIZE_TRAIN)
	for epoch in range(EPOCHS):
	    data_train_valid_features, train_valid_labels_all = shuffle(data_train_valid_features, train_valid_labels_all, random_state=0)
	    #data_train_features, train_labels_all = shuffle(data_train_features, train_labels_all, random_state=0)
	    end_subdata = 0
	     
	    val_acc_sub_data = [] 	
	    for subdata in range(int(n_subdata) ):
		start_subdata = end_subdata
		end_subdata = start_subdata + SUBDATA_SIZE_TRAIN
		train_data = data_train_valid_features[start_subdata:end_subdata, :]
		#train_data = data_train_features[start_subdata:end_subdata, :]
		train_labels = train_valid_labels_all[start_subdata:end_subdata]
		#train_labels = train_labels_all[start_subdata:end_subdata]
		train_labels_cate = keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_CLASSES)
		end = 0
		tr_list = []

		for k in range(NUMBER_OF_CLASSES):
			    start = end
			    end = start + FEATURE_DIM_PER_CLASS 
			    tr_data = train_data[:, start:end]
			    tr_list.append(tr_data.tolist())
			    #vl_data = data_valid_features[:, start:end]
			    #vl_list.append(vl_data.tolist())

		train_data_array = [np.array(tr_list[0]), np.array(tr_list[1]), np.array(tr_list[2]), \
					np.array(tr_list[3]), np.array(tr_list[4]), np.array(tr_list[5]), \
					np.array(tr_list[6])]
		hist = 	model_feature_fusion.fit(train_data_array, train_labels_cate, epochs=1, validation_split =0.33, shuffle=True, \
				callbacks=callbacks_list, verbose=1, batch_size=BATCH_SIZE)
	    
	#	if train:
	#		print '--------------------------------------------------------------------------------------------------------------'
	#		print '************************ Training data %d samples, epoch %d of %d, subdata %d of %d ***********************' \
	#						%(train_data.shape[0], epoch, EPOCHS, subdata, (int(n_subdata) - 1))
	#		print '--------------------------------------------------------------------------------------------------------------'
	#		hist = 	model_feature_fusion.fit(train_data_array, train_labels_cate, epochs=1, validation_split =0.33, shuffle=True, \
	#			callbacks=callbacks_list, verbose=1, batch_size=BATCH_SIZE)
	#		val_acc = hist.history['val_acc']
	#		val_acc_sub_data.append(val_acc)
	#	else:
	#		print "Not training epoch %d "%epoch
	#    if train:
	#	    val_acc_sub_data = np.asarray(val_acc_sub_data)
	#    mean_val_acc_full_data = np.mean(val_acc_sub_data.flatten())
	#	    mean_val_acc_full_data_history.append(mean_val_acc_full_data)
	#	    train = function_is_train(mean_val_acc_full_data_history)
	#    #print val_acc

'''
-----------------------------------------------------------------------------------------------------------
                                                    TESTING
----------------------------------------------------M-------------------------------------------------------
                                                   
'''

end_subdata = 0
n_subdata = np.ceil(test_labels_all.shape[0] / SUBDATA_SIZE_TEST)
accuracy_array = []

for subdata in range(int(n_subdata)):
    start_subdata = end_subdata
    end_subdata = start_subdata + SUBDATA_SIZE_TEST
    test_data = data_test_features[start_subdata:end_subdata, :]
    test_labels = test_labels_all[start_subdata:end_subdata]
    test_labels_cate = keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_CLASSES)
    print '******* Testing data %d samples, subdata %d of %d *******'%(test_data.shape[0], subdata, (int(n_subdata) - 1))
    ts_list = []
    end = 0
    for k in range(NUMBER_OF_CLASSES):
                start = end
                end = start + FEATURE_DIM_PER_CLASS
                ts_data = test_data[:, start:end]
                ts_list.append(ts_data.tolist())

    print "Building test input"
    test_data_array = [np.array(ts_list[0]),  np.array(ts_list[1]), \
                      np.array(ts_list[2]), np.array(ts_list[3]),\
                       np.array(ts_list[4]), np.array(ts_list[5]), np.array(ts_list[6])]

    score = model_feature_fusion.evaluate(test_data_array, test_labels_cate, batch_size=4)
    prob = model_feature_fusion.predict(test_data_array, batch_size=4)
    pdb.set_trace()
    filename = path_cross_features + 'for_confusion_mat_' + str(subdata) + '_'
    scipy.io.savemat(filename, dict(class_prob = prob, gt_test_labels = test_labels_cate))	
    print "**********"
    print("%s: %.2f%%" % (model_feature_fusion.metrics_names[1], score[1]*100))
    #print("%s: %.2f" % (model_feature_fusion.metrics_names[0], score[0]))
    accuracy_array.append(score[1]*100)

accuracy_array = np.asarray(accuracy_array)
MAE = np.mean(accuracy_array)
print "------------------------------------------------------------------"
print "------------------------------------------------------------------"
print "Dataset: %s"%dataset_name
print "Data path: %s"%path_cross_features
print "Feature dim: %d"%dim_feature
print "Method: %s, Accuracy MLP :%f " %(method, MAE)
print "Method: %s, Accuracy SVM :%f " %(method, 100*accuracy_svm)
print "Method: %s, Accuracy KNN :%f " %(method, 100*accuracy_knn)
print "------------------------------------------------------------------"
print "------------------------------------------------------------------"


# In[ ]:




