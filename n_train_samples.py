import numpy as np
import os
import scipy.io


n_train_samples = np.array([ 222,  184,  329,  271,  345,   90,  613,  228,  546,  118,  134,
        290,  183,  178, 3042,  261,  140,  151,  105,  179,   83,  111,
         97,  123,  239,  146,   30,  135,  124,  193,  209,   91])
n_test_samples = np.array([74, 61, 109, 90, 115, 30, 204, 76, 182, 39, 44, 96, 61, 59, 1014, 87, 46, 50, 35, 59, 27, 37, 32, 41, 79, 48, 10, 45, 41, 64, 69, 30])

n_train_classes = np.size(n_train_samples)
print(n_train_classes)

cnt = 0
for i in range(n_train_classes):
	classI_labels_array_train = np.empty(n_train_samples[i])
	classI_labels_array_train.fill(i+1)
	classI_labels_array_test = np.empty(n_test_samples[i])
	classI_labels_array_test.fill(i+1)
	print (i+1)	
	if cnt == 0:
		cross_features_all_classes_labels_train = classI_labels_array_train
		cross_features_all_classes_labels_test = classI_labels_array_test
		cnt = 1
	else:
		cross_features_all_classes_labels_train = np.hstack((cross_features_all_classes_labels_train, classI_labels_array_train))
        	cross_features_all_classes_labels_test = np.hstack((cross_features_all_classes_labels_test, classI_labels_array_test))
DATA_SAVE_PATH = '/home/SharedData/omkar/'

file_name = DATA_SAVE_PATH + 'data/' + 'apy' + '_'+ '300' + '_'+ '1000' + '_cc1_data_part_cross_feat_ALL_CLASS_tr_labels' + '.mat'
scipy.io.savemat(file_name, dict(cross_feautures_all_classes_tr_labels = cross_features_all_classes_labels_train))

file_name = DATA_SAVE_PATH + 'data/' + 'apy' + '_'+ '300' + '_'+ '1000' + '_cc1_data_part_cross_feat_ALL_CLASS_ts_labels' + '.mat'
scipy.io.savemat(file_name, dict(cross_feautures_all_classes_ts_labels = cross_features_all_classes_labels_test))


