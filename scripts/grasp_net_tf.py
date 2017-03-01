#!/usr/bin/env python
import rospy
import glob, os
import cv2
import numpy as np
import math as m
import random

from utils import *
from dataset_reader import *
from config import *

import tensorflow as tf
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
from imutils import rotate


class graspNetTf():
	# parameters:
	# channels: specifies the type of data being fed to the NN
	#			7: assumes (RGB, Normal, Depth) 
	#			5: assumes (Gray, Normal, Depth) 
	def __init__(self, paramfile=None, channels=5):
		with tf.device('/gpu:0'):

			print "Initializing Tensorflow network"
			config = tf.ConfigProto()
			config.allow_soft_placement=True
			config.gpu_options.allocator_type = 'BFC'
			self.sess = tf.InteractiveSession(config=config)

			self.input_shape = [None, channels, X_H, X_W]
			self.output_shape = [None, 2]

			self.x = tf.placeholder(tf.float32, self.input_shape, name="input_x")
			self.y = tf.placeholder(tf.float32, self.output_shape, name="input_y")
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

			self.layer1_in = tf.reshape(self.x, [-1,X_W,X_H,channels])
			print "Input:", self.layer1_in.get_shape()
			self.W_conv1 = self.weight_variable([7, 7, channels, 16])
			self.b_conv1 = self.bias_variable([16])
			self.h_conv1 = tf.nn.relu(self.conv2d(self.layer1_in, self.W_conv1) + self.b_conv1)
			self.h_pool1 = self.max_pool(self.h_conv1, size=2)

			print "Pool1:", self.h_pool1.get_shape()

			self.layer2_in = self.h_pool1
			self.W_conv2 = self.weight_variable([5, 5, 16, 32])
			self.b_conv2 = self.bias_variable([32])
			self.h_conv2 = tf.nn.relu(self.conv2d(self.layer2_in, self.W_conv2) + self.b_conv2)
			self.h_pool2 = self.max_pool(self.h_conv2, size=1)

			print "Pool2:", self.h_pool2.get_shape()

			self.layer3_in = self.h_pool2
			self.W_conv3 = self.weight_variable([3, 3, 32, 64])
			self.b_conv3 = self.bias_variable([64])
			self.h_conv3 = tf.nn.relu(self.conv2d(self.layer3_in, self.W_conv3) + self.b_conv3)

			print "Conv3:", self.h_conv3.get_shape()

			self.layer4_in = self.h_conv3
			self.W_conv4 = self.weight_variable([3, 3, 64, 64])
			self.b_conv4 = self.bias_variable([64])
			self.h_conv4 = tf.nn.relu(self.conv2d(self.layer4_in, self.W_conv4) + self.b_conv4)

			print "Conv4:", self.h_conv4.get_shape()

			self.layer5_in = self.h_conv4
			self.W_conv5 = self.weight_variable([3, 3, 64, 64])
			self.b_conv5 = self.bias_variable([64])
			self.h_conv5 = tf.nn.relu(self.conv2d(self.layer5_in, self.W_conv5) + self.b_conv5)
			self.h_pool5 = self.max_pool(self.h_conv5, size=2)
			
			print "Pool5:", self.h_pool5.get_shape()

			self.W_fc1 = self.weight_variable([8*4*64, 256])
			self.b_fc1 = self.bias_variable([256])
			self.h_pool2_flat = tf.reshape(self.h_pool5, [-1, 8*4*64])
			self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
			self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.dropout_keep_prob)

			print "FC1:", self.h_fc1.get_shape()

			self.W_fc2 = self.weight_variable([256, 2])
			self.b_fc2 = self.bias_variable([2])
			self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

			print "Output:", self.y_conv.get_shape()

			self.saver = tf.train.Saver({'w1':self.W_conv1, 'w2':self.W_conv2, 'w3s':self.W_conv3, \
										 'w4':self.W_conv4, 'w5':self.W_conv5, 'wfc1':self.W_fc1,  \
										 'wfc2':self.W_fc2, 'b1':self.b_conv1, 'b2':self.b_conv2, \
										 'b3':self.b_conv3, 'b4':self.b_conv4, 'b5':self.b_conv5, \
										 'bfc1':self.b_fc1, 'bfc2':self.b_fc2})

			if paramfile is not None:
				self.load_params(paramfile)

			self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y))
			self.train_step = tf.train.AdamOptimizer(2e-4).minimize(self.cross_entropy)
			self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y,1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
			
				
	def train(self, xin_train, yout_train, xin_test, yout_test):

		plt.ion()
		tf.initialize_all_variables().run()
		
		train_accuracy_list = []
		train_accuracy_list_avg = []
		test_accuracy_list = []
		epochs = []

		train_minibatch_size = 1759
		test_minibatch_size = xin_test.shape[0]

		if xin_train.shape[0]%train_minibatch_size != 0:
			print "Not an even batch size for training data"
		if xin_test.shape[0]%test_minibatch_size != 0:
			print "Not an even batch size for test data"

		num_train_minibatch = (xin_train.shape[0]/train_minibatch_size)
		num_test_minibatch = (xin_test.shape[0]/test_minibatch_size)

		print "Number of training minibatches:", num_train_minibatch
		print "Number of test minibatches:", num_test_minibatch

		num_iter = 400
		for i in range(num_iter):

			# compute test accuracy in batches
			test_accuracy_sum = 0.0
			for k in range(num_test_minibatch):
				test_batch_start =  test_minibatch_size * k
				test_batch_end = min(test_minibatch_size * (k+1), xin_test.shape[0])
				test_batch = xin_test[test_batch_start:test_batch_end]
				test_labels = yout_test[test_batch_start:test_batch_end]

				#print "SHAPE", test_batch.shape, test_labels.shape
				self.test_accuracy = self.accuracy.eval(feed_dict={self.x:test_batch, self.y:test_labels, self.dropout_keep_prob:1.0})
				print("test accuracy, batch %d = %f after %d iterations"%(k, self.test_accuracy, i))
				test_accuracy_sum += self.test_accuracy

			average_test_accuracy = test_accuracy_sum/float(num_test_minibatch)
			test_accuracy_list.append(average_test_accuracy)
			print "average test accuracy:", average_test_accuracy

						# run training iterations
			print "epoch:", i
			for j in range(num_train_minibatch):
				
				batch_start =  train_minibatch_size * j
				batch_end = min(train_minibatch_size * (j+1), xin_train.shape[0])
				batch = xin_train[batch_start:batch_end]
				labels = yout_train[batch_start:batch_end]

				#print "SHAPE", batch.shape, labels.shape
				print "Batch", j, '-- datapoints:', str(batch_start)+'-'+str(batch_end), '-- size:',  batch.shape

				self.train_accuracy = self.accuracy.eval(feed_dict={self.x:batch, self.y:labels, self.dropout_keep_prob:0.5})
				print("epoch %d, training accuracy %f"%(i, self.train_accuracy))
				self.train_step.run(feed_dict={self.x:batch, self.y:labels, self.dropout_keep_prob:0.5})
				train_accuracy_list.append(self.train_accuracy)
			
			# avgerage the training accuracies for the epoch
			train_accuracy_avg = sum(train_accuracy_list)/len(train_accuracy_list)
			train_accuracy_list_avg.append(train_accuracy_avg)
			
			# for plotting
			epochs.append(i)
			print len(epochs), len(test_accuracy_list)
			
			plt.pause(0.05)
			plt.plot(epochs, test_accuracy_list, 'r-')
			plt.title('Accuracy of Grasping Network During Training')
			plt.xlabel('epoch', fontsize=14, color='black')
			plt.ylabel('accuracy (test dataset)', fontsize=14, color='black')
			plt.show()
			plt.savefig(DIR_PROJ+"graspnet_accuracy.png")
			
				
			save_path = self.saver.save(self.sess, DIR_PROJ+"tf_grasp_model.ckpt")

	
	def load_params(self, fname):
		self.saver.restore(self.sess, DIR_PROJ+"tf_grasp_model.ckpt")
		print "Model restored"

	def eval(self, x, y):
		self.result = self.y_conv 
		return self.sess.run(self.y_conv, feed_dict={self.x:x, self.dropout_keep_prob:1.0}), self.sess.run(self.correct_prediction, feed_dict={self.x:x, self.y:y, self.dropout_keep_prob:1.0})





	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
  		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool(self, x, size):
  		return tf.nn.max_pool(x, ksize=[1, size, size, 1], \
                        	  strides=[1, size, size, 1], padding='SAME')


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']





"""
# this number MUST be #pos + #negative samples before data augmentations (flip, rotate etc)
num_datapoints = 19544
xin, yout = makeCARRTBatches(num_datapoints, save_np_file=DIR_PROJ+'carrt_dataset', verbose=False)
"""
"""
#xin, yout = makeCornellBatches(num_datapoints, batchsize = 50, verbose=True, lr='left', num_rotations=5, save_np_file=None)
"""

"""
print get_available_gpus()

param_file = DIR_PROJ+"/params1"
xin = np.load(DIR_PROJ+'carrt_dataset_x.npy')
yout = np.load(DIR_PROJ+'carrt_dataset_y.npy')

print "Success reading numpy dataset:"
print "X: size", xin.shape
print "Y: size", yout.shape

pos = 0
neg = 0
for y in yout:
	if y[0] == 1:
		pos += 1
	else:
		neg += 1


print ">> Positive samples:", pos
print ">> Negative samples:", neg
# resize and normalize input
xin = npResizeImgs(xin, X_W, X_H)

xin[:,4] = normalizeInput(xin[:,4])
xin[:,0] = normalizeInput(xin[:,0])
xin[:,1:3] = normalizeInput(xin[:,1:3])
xin = normalizeInput(xin)
#xin = whitenInput(xin)



xin_train = xin
yout_train = yout
xin_test = xin_train.copy()
yout_test = yout_train.copy()

percent_test = 10
test_set = [xin_test, yout_test]
test_set = shuffle_in_unison(test_set)

# remove some datapoints for validation
xin_test = test_set[0][0:int(test_set[0].shape[0]*(percent_test)*0.01)]
yout_test = test_set[1][0:int(test_set[1].shape[0]*(percent_test)*0.01)]

xin_train = test_set[0][int(test_set[0].shape[0]*(percent_test)*0.01):]
yout_train = test_set[1][int(test_set[1].shape[0]*(percent_test)*0.01):]

print "input size:", xin_train.shape, "output size:", yout_train.shape
print "test input size:", xin_test.shape, "test output size:", yout_test.shape
#visualizeTrainingSamples(gray, names, xin, yout, rects, 8)

filter_file = DIR_PROJ+'/filters/filter'
out_param_file = DIR_PROJ+"params/param_file_carrt_i166_e200_d00"
param_file = DIR_PROJ+"pretrain/vgg_cnn/vgg_cnn_s.pkl"

graspNet = graspNetTf()
graspNet.train(xin_train, yout_train, xin_test, yout_test)
"""