#This is the script to prepare data and labels for stanford 40 dataset with 9532 samples.

import re
import glob
import numpy as np
import pdb
import scipy.io

BASE_PATH = "/nfs4/omkar/Documents/study/phd-research/data/dataset/stanford-action-40/ImageSplits/"

#Number of samples per class
filename = BASE_PATH + 'actions_original.txt'
infile = open(filename,"r")
lines = infile.readlines()
number_of_samples_per_class = np.array([], dtype=int)
for line in lines:
	print line
	a=re.findall(r'\d+', line)
	if not a:
		print 'No number found in this line'
	else:
		#print a[0]
		number_of_samples_per_class = np.hstack((number_of_samples_per_class, int(a[0])))

cum_sum_n_samples = np.cumsum(number_of_samples_per_class, dtype=int)
offset_n_samples = np.array([0])
offset_n_samples = np.hstack((offset_n_samples, cum_sum_n_samples))
print number_of_samples_per_class
print cum_sum_n_samples
print offset_n_samples

filename = BASE_PATH + 'actions_names.txt'
with open(filename) as f:
    action_names = f.read().splitlines()

#Dataset labels...........................
class_labels = np.arange(1, 41, 1, dtype=int)
dataset_labels = np.array([], dtype=int)
for k in class_labels:
	this_class_labels = k * np.ones(number_of_samples_per_class[k -1],)
	dataset_labels = np.hstack((dataset_labels, this_class_labels))

#Training..................................
cnt = 0
k = 0

for fl in action_names:
	filename = BASE_PATH + str(fl) + '_train.txt' 
	print "%d %s"%(k+1, filename)
	infile = open(filename,"r")
	lines = infile.readlines()
	train_sample_indices_this_class = np.array([], dtype=int)
	for line in lines:
		#print line
		a=re.findall(r'\d+', line)
		#print a[0]
		train_sample_indices_this_class = np.hstack((train_sample_indices_this_class, (int(a[0]) + offset_n_samples[k])))

	if cnt ==0:
		cnt = 1
		train_sample_indices = np.asarray(train_sample_indices_this_class)
	else:
		train_sample_indices = np.hstack((train_sample_indices, train_sample_indices_this_class))
	k = k + 1
	pdb.set_trace()

print train_sample_indices.shape	

tmp1 = np.zeros(dataset_labels.shape[0], dtype=int)
tmp1[train_sample_indices -1] = 1
print np.sum(tmp1)
dataset_train_labels = np.multiply(tmp1, dataset_labels)
train_sample_indices = np.reshape(train_sample_indices, (1, train_sample_indices.shape[0]))
print train_sample_indices.shape

#Testing..................................
test_sample_indices = []
cnt = 0
k = 0
for fl in action_names:
	filename = BASE_PATH + str(fl) + '_test.txt' 
	print "%d %s"%(k+1, filename)
	infile = open(filename,"r")
	lines = infile.readlines()
	test_sample_indices_this_class = np.array([], dtype=int)
	for line in lines:
		#print line
		a=re.findall(r'\d+', line)
		#print a[0]
		test_sample_indices_this_class = np.hstack((test_sample_indices_this_class,(int(a[0]) + offset_n_samples[k])))

	if cnt ==0:
		cnt = 1
		test_sample_indices = np.asarray(test_sample_indices_this_class)
	else:
		test_sample_indices = np.hstack((test_sample_indices, test_sample_indices_this_class))
	k = k+1

print test_sample_indices.shape	
tmp2 = np.zeros(dataset_labels.shape[0], dtype=int)
tmp2[test_sample_indices -1] = 1
print np.sum(tmp2)
dataset_test_labels = np.multiply(tmp2, dataset_labels)
test_sample_indices = np.reshape(test_sample_indices, (1, test_sample_indices.shape[0]))
print test_sample_indices.shape	

dataset_labels = np.reshape(dataset_labels, (1, dataset_labels.shape[0]))
dataset_train_labels = np.reshape(dataset_train_labels, (1, dataset_train_labels.shape[0]))
dataset_test_labels = np.reshape(dataset_test_labels, (1, dataset_test_labels.shape[0]))

filename = 'stanford40_dataset_labels'
scipy.io.savemat(filename, dict(dataset_labels = dataset_labels, \
 			   dataset_train_labels = dataset_train_labels, \
			   dataset_test_labels = dataset_test_labels, \
			   final_train_indices = train_sample_indices, \
			   final_test_indices = test_sample_indices))
print filename

print test_sample_indices.shape	
print train_sample_indices.shape	
print dataset_labels.shape	
print dataset_train_labels.shape	
print dataset_test_labels.shape	
