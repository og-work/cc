import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


SAVE = 1
tmp = tf.Variable([], name="tmp")
if SAVE:
	v1 = tf.Variable([1., 1] , name="v1")
	v2 = tf.Variable([2., 2], name="v2")
	v3 = tf.Variable([3.,3] , name="v3")
	v4 = tf.Variable([4.,4] , name="v4")
else:
        tmp = scipy.io.loadmat('v1_value')
	init_v1 = tmp['v1_value']
        tmp = scipy.io.loadmat('v2_value')
	init_v2 = tmp['v2_value']
	v1 = tf.Variable(init_v1 , name="v1")
	v2 = tf.Variable(init_v2 , name="v2")
	v3 = tf.Variable([33., 33] , name="v3")
	v4 = tf.Variable([44., 44] , name="v4")

a = tf.add(v1, v2)
b = tf.add(a, v3)
c = tf.add(b, v4)

if SAVE == 1:
	print "...Saving"
	# Let's create a Saver object
	# By default, the Saver handles every Variables related to the default graph
	variables = slim.get_variables_to_restore()
	variables_to_restore = [v1, v2]
	v12_saver = tf.train.Saver(variables_to_restore)
	all_saver = tf.train.Saver() 
	# But you can precise which vars you want to save under which name
	#v12_saver = tf.train.Saver([v1, v2]) 
	#v34_saver = tf.train.Saver([v3, v4])

	with tf.Session() as sess:
		# Init v and v2   
		sess.run(tf.global_variables_initializer())
		# Now v1 holds the value 1.0 and v2 holds the value 2.0
		# We can now save all those values
		sess.run(a)
		all_saver.save(sess, 'm4/data-all', global_step=1)
		v12_saver.save(sess, 'm5/data-v12', global_step=1) 
		#v34_saver.save(sess, 'm5/data-v34', global_step=1) 
		print sess.run(a)
		print sess.run(b)
		print sess.run(c)
		print v1
		print tmp
		v1_value = v1.eval()
		v2_value = v2.eval()
		print type(v1_value)
		filename = 'v1_value'
		scipy.io.savemat(filename, dict(v1_value = v1_value))
		filename = 'v2_value'
		scipy.io.savemat(filename, dict(v2_value = v2_value))

if SAVE == 0:
	print "...loading"
#	loader34 = tf.train.import_meta_graph('m5/data-v34.meta')

	# We can now access the default graph where all our metadata has been loaded
	#tf.reset_default_graph()
	#graph = tf.get_default_graph()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#print sess.run(a)
		loader12 = tf.train.import_meta_graph('m5/data-v12-1.meta')
		loader12.restore(sess, tf.train.latest_checkpoint('m5/'))	
		print 'v1', v1.eval()
		print sess.run(a)
		#sess.run(tf.global_variables_initializer())
#		loader34.restore(sess, tf.train.latest_checkpoint('m5/'))	
		print sess.run(b)
		#print sess.run(b)
		#print sess.run(c)
