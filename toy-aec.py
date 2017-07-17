import tensorflow as tf
import numpy as np
import math
#import pandas as pd
#import sys


input = np.array([[2.0, 1.0, 1.0, 2.0],
                 [-2.0, 1.0, -1.0, 2.0],
                 [0.0, 1.0, 0.0, 2.0],
                 [0.0, -1.0, 0.0, -2.0],
                 [0.0, -1.0, 0.0, -2.0]])

input_new = np.array([[12.0, 311.0, 11.0, 12.0],
                 [-12.0, 11.0, -11.0, 12.0],
                 [130.0, 311.0, 410.0, 12.0],
                 [10.0, -116.0, 109.0, -152.0],
                 [210.0, -131.0, 170.0, -182.0]])
# Code here for importing data from file

noisy_input = input + .2 * np.random.random_sample((input.shape)) - .1
output = input_new

# Scale to [0,1]
scaled_input_1 = np.divide((noisy_input-noisy_input.min()), (noisy_input.max()-noisy_input.min()))
scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
# Scale to [-1,1]
scaled_input_2 = (scaled_input_1*2)-1
scaled_output_2 = (scaled_output_1*2)-1

input_data = scaled_input_2
output_data = scaled_output_2

# Autoencoder with 1 hidden layer
n_samp, input_dim = input_data.shape 
n_hidden = 2

x = tf.placeholder("float", [None, input_dim])
# Weights and biases to hidden layer
Wh = tf.Variable(tf.random_uniform((input_dim, n_hidden), -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
# Weights and biases to hidden layer
#Wo = tf.transpose(Wh) # tied weights
Wo = tf.Variable(tf.random_uniform((n_hidden, input_dim), -1.0 / math.sqrt(n_hidden), 1.0 / math.sqrt(n_hidden)))
bo = tf.Variable(tf.zeros([input_dim]))
y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
# Objective functions
y_ = tf.placeholder("float", [None,input_dim])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
meansq = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

n_rounds = 5000
batch_size = min(50, n_samp)

for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data[sample][:]
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    if i % 100 == 0:
        print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})

save_path = saver.save(sess, "data/model.ckpt")
print("Model saved in file: %s" % save_path)

print "Target:"
print output_data
print "Final activations:"
print sess.run(y, feed_dict={x: input_data})
decoded = sess.run(y, feed_dict={x: input_data})
print "Final activations: np"
print decoded
print "Final weights (input => hidden layer)"
print sess.run(Wh)
print "Final biases (input => hidden layer)"
print sess.run(bh)
print "Final weights (hidden => output)"
print sess.run(Wo)
print "Final biases (hidden layer => output)"
print sess.run(bo)
print "Final activations of hidden layer"
print sess.run(h, feed_dict={x: input_data})
acti_hidden = sess.run(h, feed_dict={x: input_data})
print "Final activations of hidden layer np"
print acti_hidden

print('*******************Restoring******************')
saver.restore(sess, "data/model.ckpt")
print("Model restored.")
print "Final weights (input => hidden layer)"
print sess.run(Wh)
