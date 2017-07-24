from train_tensorflow_cc import classifier_input, classifier_output, function_train_classifier_for_cc
import scipy.io
import pdb

PATH_DATA = '/home/SharedData/omkar/data/'
number_of_train_classes = 4

features_tr = scipy.io.loadmat(PATH_DATA + 'apy_200_800_cc1_data_part_cross_feat_ALL_CLASS_tr_4_.mat')
cross_features_all_classes_train = features_tr['cross_feautures_all_classes_tr']

features_ts = scipy.io.loadmat(PATH_DATA + 'apy_200_800_cc1_data_part_cross_feat_ALL_CLASS_ts_4_.mat')
cross_features_all_classes_test = features_ts['cross_feautures_all_classes_ts']

labels_tr = scipy.io.loadmat(PATH_DATA + 'apy_200_800_cc1_data_part_cross_feat_ALL_CLASS_tr_labels_4_.mat')
cross_features_all_classes_labels_train = labels_tr['cross_feautures_all_classes_labels_tr']

labels_ts = scipy.io.loadmat(PATH_DATA + 'apy_200_800_cc1_data_part_cross_feat_ALL_CLASS_ts_labels_4_.mat')
cross_features_all_classes_labels_test = labels_ts['cross_feautures_all_classes_labels_ts']

input_dim = (cross_features_all_classes_train.shape[1])
dimension_hidden_layer1_classifier = int(0.7 * input_dim) 
obj_classifier_input = classifier_input()
obj_classifier_input.epochs = 1
obj_classifier_input.number_of_train_classes = number_of_train_classes
obj_classifier_input.dim_hidden_layer1 = dimension_hidden_layer1_classifier
obj_classifier_input.train_data = cross_features_all_classes_train 
obj_classifier_input.train_labels = cross_features_all_classes_labels_train
obj_classifier_input.test_data = cross_features_all_classes_test
obj_classifier_input.test_labels = cross_features_all_classes_labels_test
obj_classifier_output = function_train_classifier_for_cc(obj_classifier_input)

print "Number of classes %d"%number_of_train_classes
print "Input Dim %d, Hidden layer(classifier) dim %d"%(input_dim, dimension_hidden_layer1_classifier)
print "Training data (%d, %d)"%(cross_features_all_classes_train.shape)
print "Testing data (%d, %d)"%(cross_features_all_classes_test.shape)
