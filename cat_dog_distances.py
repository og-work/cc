'''
-----------------------------------------------------
	stanford40_cat_dog_distance.py
-----------------------------------------------------
'''
import time
import numpy as np
from numpy import linalg as LA
import os
import pdb
import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tsne import tsne
import scipy.interpolate as interpolate
import matplotlib.cm as cm
import colorsys
import matplotlib.colors as clrs
'''
-----------------------------------------------------
	Data path/ macro/ constants
-----------------------------------------------------
'''

dataset_list = ['stanford40', 'willow', 'mnist', 'mit67']
dataset_name = dataset_list[1]
if dataset_name == 'stanford40':
	data_path = '/nfs4/omkar/Documents/study/phd-research/codes/tf-codes/data-stanford40/data13/'
	NUMBER_OF_CLASSES = 40
	data_save_path = './stanford40/'
	if not os.path.exists(data_save_path):
	    os.makedirs(data_save_path)

elif dataset_name == 'willow':
	data_path = '/nfs4/omkar/Documents/study/phd-research/codes/tf-codes/data-willow/data2_hemachandra/'
	NUMBER_OF_CLASSES = 7
	data_save_path = './willow/'
	if not os.path.exists(data_save_path):
	    os.makedirs(data_save_path)
else:
	data_path = '/nfs4/omkar/Documents/study/phd-research/codes/tf-codes/data-mit67/data2_hemachandra/'
	NUMBER_OF_CLASSES = 67
	data_save_path = './mit67/'
	if not os.path.exists(data_save_path):
	    os.makedirs(data_save_path)


def function_scatter_plot(input_data, output_data, decoded_data, mean_input_data, mean_output_data, classI, classJ):
	ip_data_decoded_data = np.vstack((input_data, output_data))
	ip_data_decoded_data = np.vstack((ip_data_decoded_data, decoded_data))
	ip_data_decoded_data = np.vstack((ip_data_decoded_data, mean_input_data))
	ip_data_decoded_data = np.vstack((ip_data_decoded_data, mean_output_data))
	ip_data_decoded_data_labels = np.hstack(([0]*input_data.shape[0], [1]*input_data.shape[0]))
	ip_data_decoded_data_labels = np.hstack((ip_data_decoded_data_labels, [2]*input_data.shape[0]))
	ip_data_decoded_data_labels = np.hstack((ip_data_decoded_data_labels, [3, 4]))
	
	num_data_points = input_data.shape[0]
	#Y = TSNE(n_components=2).fit_transform(ip_data_decoded_data)
	#ip_data_decoded_data = PCA(n_components=2).fit_transform(ip_data_decoded_data)
        #Y = tsne(ip_data_decoded_data, 2, 50, 20.0);
	Y = PCA(n_components=2).fit_transform(ip_data_decoded_data)
	print num_data_points, ip_data_decoded_data.shape, Y.shape
	data_names = ['Input', 'Output', 'Decoded', 'Mean IP', 'Mean OP']
	clusters = [0,1,2,3,4,5]
	colors = ['red', 'blue', 'green', 'magenta', 'cyan']
	marker_size = 70
	for data_name, color, cluster in zip(data_names, colors, clusters):
		indices = np.flatnonzero(ip_data_decoded_data_labels == cluster)
		if data_name == 'Mean IP' or data_name == 'Mean OP':
			marker_size = 200
		plt.scatter(Y[indices, 0], Y[indices, 1], label = data_name, s = marker_size, c=color);
	
	plt.legend(loc='best', fontsize=10)

	return

def function_load_data_for_different_ip_class(classI):
	filename = data_path + dataset_name + '_50_500_' + 'feat_fusion_clsfr_test_labels'
	tmp = scipy.io.loadmat(filename)
	test_labels_all = tmp['test_labels']
	test_labels_all = test_labels_all - 1

	filename = data_path + dataset_name + '_50_500_' + 'cec_features_class_ts_' + str(classI + 1)
        print "Loading %s"%filename
        tmp = scipy.io.loadmat(filename)
        data_test = tmp['cross_feautures_ts']
	
	return data_test, labels_test

'''
-----------------------------------------------------
Main
-----------------------------------------------------
'''


stats_ip_op_all_classes = []

for classI in range(NUMBER_OF_CLASSES):
	for classJ in range(NUMBER_OF_CLASSES):
		if classI != classJ:
				print '-------- Class %d and Class %d ---------'%(classI+1, classJ+1)
				filename_input = '_debug_wtinit_1_feat_fusion_clsfr500_50_' + dataset_name + '_' + str(classI + 1) + '_' + str(classJ + 1) + '_valid_input'
				filename_decoded ='_debug_wtinit_1_feat_fusion_clsfr500_50_' + dataset_name + '_' + str(classI + 1) + '_' + str(classJ + 1) + '_valid_decoded'
				filename_output = '_debug_wtinit_1_feat_fusion_clsfr500_50_' + dataset_name + '_'+ str(classI + 1) + '_' + str(classJ + 1) + '_valid_output'

				t1 = scipy.io.loadmat(data_path + filename_input)
				t2 = scipy.io.loadmat(data_path + filename_decoded)
				t3 = scipy.io.loadmat(data_path + filename_output)

				input_data = t1['debug_valid_input']
				decoded_data = t2['debug_valid_decoded']
				output_data = t3['debug_valid_output']
				
				print input_data.shape
				mean_output_data = np.mean(output_data, axis=0)
				mean_input_data = np.mean(input_data, axis=0)

				distance_ip_mean_op_mean = 100*LA.norm(mean_input_data - mean_output_data)/(np.sqrt(4*input_data.shape[1]))
				
				distance_ip_mean_op_mean_array = np.tile(distance_ip_mean_op_mean, (1, input_data.shape[0]))
				distance_ip_mean_op_mean_array = distance_ip_mean_op_mean_array.flatten()
				diff_ip_decoded = decoded_data - np.tile(mean_input_data, (input_data.shape[0], 1))
				diff_op_decoded = decoded_data - np.tile(mean_output_data, (output_data.shape[0], 1))

				#normalised distance between 0 to 1
				distance_ip_decoded = 100*LA.norm(diff_ip_decoded, axis=1)/(np.sqrt(4*input_data.shape[1]))
				distance_op_decoded = 100*LA.norm(diff_op_decoded, axis=1)/(np.sqrt(4*output_data.shape[1]))
				
				title_string = dataset_name + ': class ' + str(classI + 1) + ' and class ' + str(classJ + 1)
				plt.title(title_string)	
				plt.plot(distance_ip_decoded,'red', label = 'distance(meanIP, decoded)')
				plt.plot(distance_op_decoded,'green', label = 'distance(meanOP, decoded)')
				plt.plot(distance_ip_mean_op_mean_array, 'blue', label = 'distance(meanIP, meanOP)')
				legend = plt.legend(loc='upper right', shadow=True)
				plt.grid()
				#plt.show()
				stats_ip_op = np.zeros(distance_ip_decoded.shape[0])
				margin = 0
				stats_ip_op[distance_ip_decoded > distance_op_decoded + margin] = 1
				stats_ip_op_all_classes.append((np.sum(stats_ip_op)/stats_ip_op.shape[0]))
				
				figurename = data_save_path + 'class_' + str(classI + 1) + '_class_' + str(classJ + 1)	
				#plt.savefig(figurename + '.eps', format='eps',dpi=1000)
				plt.savefig(figurename + '.png')
				plt.close("all")

				function_scatter_plot(input_data, output_data, decoded_data, mean_input_data, mean_output_data, classI, classJ)
				plt.title(title_string)
				figurename = data_save_path + 'SCATTER_PCA_class_' + str(classI + 1) + '_class_' + str(classJ + 1)	
				#plt.savefig(figurename + '.eps', format='eps',dpi=1000)
				plt.savefig(figurename + '.png')
				#plt.show()
				#pdb.set_trace()
				plt.close("all")


stats_ip_op_all_classes = np.asarray(stats_ip_op_all_classes)
filename = data_save_path + dataset_name + '_stats_ip_op_all_classes'
scipy.io.savemat(filename, dict(stats_ip_op_all_classes = stats_ip_op_all_classes))

plt.stem(stats_ip_op_all_classes*100)
plt.ylabel('% of decoded samples close to mean of output class samples')
plt.title('Dataset:' + dataset_name)
plt.grid()
#plt.show()
figurename = data_save_path + dataset_name + '_stats_ip_op_all_classes_stems'	
plt.savefig(figurename + '.eps', format='eps',dpi=1000)
plt.savefig(figurename + '.png')


plt.close("all")
plt.plot(stats_ip_op_all_classes*100)
plt.ylabel('% of decoded samples close to mean of output class samples')
plt.title('Dataset:' + dataset_name)
plt.grid()
#plt.show()
figurename = data_save_path + dataset_name + '_stats_ip_op_all_classes_plot'	
plt.savefig(figurename + '.eps', format='eps',dpi=1000)
plt.savefig(figurename + '.png')



	 

