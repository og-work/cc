#This is the script to prepare data and labels for oxford pets dataset with ???? samples.

import re
import glob
import numpy as np
import pdb
import scipy.io
import os
import os.path
from shutil import copyfile
import xml.etree.ElementTree

import xml.etree.ElementTree as ET
tree = ET.parse('/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/annotations/xmls/Abyssinian_1.xml')
root = tree.getroot()
pdb.set_trace()
for country in root.findall('bndbox'):
     rank = country.find('xmin').text
     name = country.get('ymin')
     print name, rank


BASE_PATH = "/nfs4/omkar/Documents/study/phd-research/codes/tf-codes/codes-oxford-pets/"
IMAGE_PATH = "/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/images/"
TRAIN_DIR_PATH = "/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/finetune/train/"
VALID_DIR_PATH = "/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/finetune/valid/"
TEST_DIR_PATH = "/nfs4/omkar/Documents/study/phd-research/data/datasets/oxford-pets/finetune/test/"
TOTAL_SAMPLES = 7349
NUMBER_OF_CLASSES = 37


filename = BASE_PATH + 'train_valid_sample_names.txt'
infile = open(filename,"r")
lines = infile.readlines()
number_of_train_valid_samples_per_class = []
number_of_test_samples_per_class = []
number_of_train_valid_samples = len(lines)
class_names_array_train = ['Abyssinian']
train_dir_name = TRAIN_DIR_PATH + 'class' + str(0).zfill(3)
valid_dir_name = VALID_DIR_PATH + 'class' + str(0).zfill(3)
if not os.path.exists(train_dir_name):
	os.makedirs(train_dir_name)
if not os.path.exists(valid_dir_name):
	os.makedirs(valid_dir_name)

print number_of_train_valid_samples

number_of_train_samples_array = np.zeros(NUMBER_OF_CLASSES)
label = 0
m = 0
if 1:
	for line in lines:
		line1 = line.strip('\n')
		sample_name = line1 + '.jpg'
		line1 = line1.split('_')
		class_name = line1[0]
		print line1
		#create class name
		for k in range(len(line1) - 2):
			class_name = class_name + '_' + line1[k + 1]
		if not class_name in class_names_array_train:
			class_names_array_train.append(class_name)
			label = label + 1;
			print sample_name, class_name, label	
			train_dir_name = TRAIN_DIR_PATH + 'class' + str(label).zfill(3)
			valid_dir_name = VALID_DIR_PATH + 'class' + str(label).zfill(3)
			if not os.path.exists(train_dir_name):
				os.makedirs(train_dir_name)
				os.makedirs(valid_dir_name)
			
		src = IMAGE_PATH + sample_name
		ind = class_names_array_train.index(class_name)
		number_of_train_samples_array[ind] = number_of_train_samples_array[ind] + 1
		if 0:#number_of_train_samples_array[ind] > 69:
			dest = VALID_DIR_PATH + '/class' + str(ind).zfill(3) + '/' + sample_name
		else:
			dest = TRAIN_DIR_PATH + '/class' + str(ind).zfill(3) + '/' + sample_name
		copyfile(src, dest)	
	
filename1 = BASE_PATH + 'test_sample_names.txt'
infile1 = open(filename1,"r")
lines1 = infile1.readlines()
number_of_samples_per_class = np.array([], dtype=int)
number_of_test_samples = len(lines1)
class_names_array_test = ['Abyssinian']
test_dir_name = TEST_DIR_PATH + 'class' + str(0).zfill(3)
if not os.path.exists(test_dir_name):
	os.makedirs(test_dir_name)
print number_of_test_samples
label = 0
m = 0

if 1:
	for line in lines1:
		line1 = line.strip('\n')
		sample_number = re.findall(r'\d+', line1)
		sample_name = line1 + '.jpg'
		line1 = line1.split('_')
		class_name = line1[0]
		print line1
		#create class name
		for k in range(len(line1) - 2):
			class_name = class_name + '_' + line1[k + 1]
		if not class_name in class_names_array_test:
			class_names_array_test.append(class_name)
			label = label + 1;
			print sample_name, class_name, label	
			test_dir_name = TEST_DIR_PATH + 'class' + str(label).zfill(3)
			if not os.path.exists(test_dir_name):
				os.makedirs(test_dir_name)
			
		src = IMAGE_PATH + sample_name
		ind = class_names_array_test.index(class_name)
		dest = TEST_DIR_PATH + '/class' + str(ind).zfill(3) + '/' + sample_name
		copyfile(src, dest)	

dataset_train_labels = np.array([])
dataset_test_labels = np.array([])
for n in range(NUMBER_OF_CLASSES):
	subdir = 'class' + str(n).zfill(3)
	num_files = len([f for f in os.listdir(TRAIN_DIR_PATH + subdir)if os.path.isfile(os.path.join(TRAIN_DIR_PATH + subdir, f))])
	number_of_train_valid_samples_per_class.append(num_files)
	dataset_train_labels = np.hstack((dataset_train_labels, [n] * num_files))
	print TRAIN_DIR_PATH + subdir, num_files
	num_files = len([f for f in os.listdir(TEST_DIR_PATH + subdir)if os.path.isfile(os.path.join(TEST_DIR_PATH + subdir, f))])
	dataset_test_labels = np.hstack((dataset_test_labels, [n] * num_files))
	print dataset_test_labels
	number_of_test_samples_per_class.append(num_files)
	print TEST_DIR_PATH + subdir, num_files

np.asarray(number_of_train_valid_samples_per_class)
np.asarray(number_of_test_samples_per_class)
number_of_train_samples = np.sum(number_of_train_valid_samples_per_class)
number_of_test_samples = np.sum(number_of_test_samples_per_class)
dummy_end = [-1] * number_of_test_samples
dummy_start = [-1] * number_of_train_samples
dataset_train_labels = np.hstack((dataset_train_labels, dummy_end))
dataset_test_labels = np.hstack((dummy_start, dataset_test_labels))
print dataset_train_labels.shape, dataset_test_labels.shape

print number_of_train_samples, number_of_test_samples, (number_of_test_samples + number_of_train_samples)

#print sample_name
#	a=re.findall(r'\d+', line)
#	if not a:
#		print 'No number found in this line'
#	else:
#		#print a[0]
#		number_of_samples_per_class = np.hstack((number_of_samples_per_class, int(a[0])))
pdb.set_trace()
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
