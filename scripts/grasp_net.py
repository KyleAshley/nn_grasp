#!/usr/bin/env python
import rospy
import glob, os
import cv2
import numpy as np
import math as m
#import cv
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import nolearn

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from lasagne import layers
import math

import random

from utils import *

import theano
import numpy as np
import theano.tensor as T
import gzip
import pickle
import lasagne as L
import time

import matplotlib.pyplot as plt
from imutils import rotate
from dataset_reader import *
import cv2

from config import *

# REFERENCES:
# Implementation of Deepnetwork with:
#   0) Fully Connected Network
#	1) Convolutional deep network

# http://deeplearning.net/tutorial/rbm.
# http://deeplearning.net/tutorial/dA.html?highlight=autoencoder
# https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
# http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
# http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
# https://groups.google.com/forum/#!topic/lasagne-users/aynhS_hricE

#http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu
# DO THIS sudo ldconfig /usr/local/cuda-7.5/lib64
# https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet
# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf



# randomly show <num> training samples for the network
def visualizeTrainingSamples(imgs, names, xin, yout, rects, num=5):
	yout = np.squeeze(yout)
	names = np.squeeze(names)
	print rects.shape
	for i in range(num):
		idx = random.randint(0, xin.shape[0])
		x = xin[idx][0]
		orig = cv2.imread(str(names[idx]))
		drawing = orig.copy()
		rect_points = np.array(np.int0(rects[idx]))
		rectangle = cv2.minAreaRect(rect_points)
		box = cv2.boxPoints(rectangle)
		box = np.int0(box)
		cv2.drawContours(drawing,[box],0,(0,0,255),2)

		cv2.imshow('orig', orig)
		cv2.imshow('img', drawing)
		cv2.imshow('x', x)
		cv2.waitKey(-1)


class graspNet():

	def __init__(self, param_file=None):
		net_divider = 1.0

		self.layers =[
			(L.layers.InputLayer, 					
					{'shape':(None, 7, X_H, X_W),
					 'name': 'input'}), 			
			(L.layers.Conv2DLayer, 					
					{'num_filters':96, 							
					'stride':1, 		
					'pad':3,						
					'filter_size':(7,7), 						
					'nonlinearity':L.nonlinearities.rectify,	
					'flip_filters':False,
					'name': 'conv0'}),						
			(L.layers.DropoutLayer,					
					{'p':0.0}),									
			(L.layers.MaxPool2DLayer,					
					{'pool_size':3,								
					'ignore_border':False}),					
			(L.layers.Conv2DLayer,						
					{'num_filters':256,							
					'stride':1,									
					'filter_size':(5,5),						
					'nonlinearity':L.nonlinearities.rectify,	
					'flip_filters':False,
					'name': 'conv1'}),						
			(L.layers.DropoutLayer,					
					{'p':0.0}),									
			(L.layers.MaxPool2DLayer,					
					{'pool_size':2,								
					'ignore_border':False}),					
			(L.layers.Conv2DLayer,						
					{'num_filters':512,							
					'stride':1,									
					'pad':1,									
					'filter_size':(3,3),						
					'nonlinearity':L.nonlinearities.rectify,	
					'flip_filters':False,
					'name': 'conv2'}),						
			(L.layers.DropoutLayer,					
					{'p':0.0}),									
			(L.layers.Conv2DLayer,						
					{'num_filters':512,							
					'stride':1,									
					'pad':1,									
					'filter_size':(3,3),						
					'nonlinearity':L.nonlinearities.rectify,	
					'flip_filters':False,
					'name': 'conv3'}),						
			(L.layers.DropoutLayer,					
					{'p':0.0}),									
			(L.layers.Conv2DLayer,						
					{'num_filters':512,							
					'stride':1,									
					'pad':1,									
					'filter_size':(3,3),						
					'nonlinearity':L.nonlinearities.rectify,	
					'flip_filters':False,
					'name': 'conv4'}),						
			(L.layers.DropoutLayer,					
					{'p':0.0}),									
			(L.layers.MaxPool2DLayer,					
					{'pool_size':3,								
					'ignore_border':False}),					
			(L.layers.DenseLayer,						
					{'num_units':4096,							
					'nonlinearity':L.nonlinearities.sigmoid,
					'name': 'dense0'}),	
			(L.layers.DropoutLayer,					
					{'p':0.0}),									
			(L.layers.DenseLayer,						
					{'num_units':4096,							
					'nonlinearity':L.nonlinearities.sigmoid,
					'name': 'dense1'}),	
			(L.layers.DenseLayer,						
					{'num_units':1,								
					'nonlinearity':L.nonlinearities.sigmoid}),	
			]


		self.net = NeuralNet(layers=self.layers,
						update_learning_rate=0.015,
						update=L.updates.nesterov_momentum,
						update_momentum=0.9,
						#update=L.updates.sgd,
						regression=True,
						verbose=1,
						eval_size=0.15,
						objective_loss_function=L.objectives.binary_crossentropy,
						max_epochs=200)
		
		if param_file !=None:
			print "Loading parameters from ", param_file
			self.net.load_params_from(param_file)


	def write_filters_to_file(self, fname):
		params = self.net.get_all_params()
		layer_counter = 0
		for p in params:
			print p
			layer_counter += 1
			filter_counter = 0
			weights = p.get_value()

			if len(weights.shape) > 2:
				for f in weights:
					kernel = np.asarray(f, dtype=np.float32)
					kernel = kernel*255+128
					viz = np.zeros(shape=kernel[0].shape)
					viz = cv2.resize(kernel[0],None,fx=20, fy=20, interpolation = cv2.INTER_CUBIC)
					cv2.normalize(viz, viz)
					viz = viz*50
					
					#print viz
					#cv2.imshow('ha', viz)
					#cv2.waitKey(-1)

					viz = viz/50
					viz = viz*12000
					viz = viz
					#print viz
					cv2.imwrite(fname+"_"+str(layer_counter)+"_"+str(filter_counter)+".png", viz)
					filter_counter += 1

	# evaluates the input array over the neural net
	def eval(self, x):
		y = np.zeros((x.shape[0],))
		for i in range(x.shape[0]):
			pred = self.net.predict(np.array([x[i]]))
			y[i] = pred
			print i, pred

		return y


	# train the network in input (x,y)
	# param_file: input parameter file (perhaps from previous trainings)
	# out_param_file: output path to write resulting params to
	# filter_file: output path to write images of filters for visualization
	# load_params: boolean flag, when set: 1 loads input parameters from <param_file>
	# pretrain: boolean flag, when set: 1 loads parameters from pretrained network

	def train(self, X, y, param_file=None, out_param_file=None, filter_file=None, load_params=False, pretrain=False):
		
		if pretrain:
			params = createNoLearnParams(param_file)
			print "Parameters", params[1].shape
			conv0_W = np.concatenate((params[0],params[0]),axis=1)
			conv0_W = np.concatenate((conv0_W,params[0]),axis=1)
			conv0_W = conv0_W[:,:7,:,:]

			conv0_b = np.concatenate((params[1],params[1]),axis=0)
			conv0_b = np.concatenate((conv0_b,params[1]),axis=0)
			conv0_b = conv0_b[:96]

			conv1_W = np.concatenate((params[2],params[2]),axis=1)
			conv1_W = np.concatenate((conv1_W,params[2]),axis=1)
			conv1_W = conv1_W[:,:96,:,:]

			conv1_b = np.concatenate((params[3],params[3]),axis=0)
			conv1_b = np.concatenate((conv1_b,params[3]),axis=0)
			conv1_b = conv1_b[:256]

			conv2_W = np.concatenate((params[4],params[4]),axis=1)
			conv2_W = np.concatenate((conv2_W,params[4]),axis=1)
			conv2_W = conv2_W[:,:256,:,:]

			conv2_b = np.concatenate((params[5],params[5]),axis=0)
			conv2_b = np.concatenate((conv2_b,params[5]),axis=0)
			conv2_b = conv2_b[:512]

			conv3_W = np.concatenate((params[6],params[6]),axis=1)
			conv3_W = np.concatenate((conv3_W,params[6]),axis=1)
			conv3_W = conv3_W[:,:512,:,:]

			conv3_b = np.concatenate((params[7],params[7]),axis=0)
			conv3_b = np.concatenate((conv3_b,params[7]),axis=0)
			conv3_b = conv3_b[:512]

			conv4_W = np.concatenate((params[8],params[8]),axis=1)
			conv4_W = np.concatenate((conv4_W,params[8]),axis=1)
			conv4_W = conv4_W[:,:512,:,:]

			conv4_b = np.concatenate((params[9],params[9]),axis=0)
			conv4_b = np.concatenate((conv4_b,params[9]),axis=0)
			conv4_b = conv4_b[:512]

			dense0_W = np.concatenate((params[10],params[10]),axis=1)
			dense0_W = np.concatenate((dense0_W,params[10]),axis=1)
			dense0_W = dense0_W[:2560,:4096]

			dense0_b = np.concatenate((params[11],params[11]),axis=0)
			dense0_b = np.concatenate((dense0_b,params[11]),axis=0)
			dense0_b = dense0_b[:4096]

			dense1_W = np.concatenate((params[12],params[12]),axis=1)
			dense1_W = np.concatenate((dense1_W,params[12]),axis=1)
			dense1_W = dense1_W[:4096,:4096]

			dense1_b = np.concatenate((params[13],params[13]),axis=0)
			dense1_b = np.concatenate((dense1_b,params[13]),axis=0)
			dense1_b = dense1_b[:4096]


			

			#http://arxiv.org/pdf/1405.3531v4.pdf
			self.net = NeuralNet(layers=self.layers,
							conv0_W=np.array(conv0_W),
							conv0_b=np.array(conv0_b),
							conv1_W=np.array(conv1_W),
							conv1_b=np.array(conv1_b),
							conv2_W=np.array(conv2_W),
							conv2_b=np.array(conv2_b),
							conv3_W=np.array(conv3_W),
							conv3_b=np.array(conv3_b),
							conv4_W=np.array(conv4_W),
							conv4_b=np.array(conv4_b),
							dense0_W=np.array(dense0_W),
							dense0_b=np.array(dense0_b),
							dense1_W=np.array(dense1_W),
							dense1_b=np.array(dense1_b),
							update_learning_rate=0.015,
							update=L.updates.nesterov_momentum,
							update_momentum=0.9,
							#update=L.updates.sgd,
							regression=True,
							verbose=1,
							eval_size=0.15,
							objective_loss_function=L.objectives.binary_crossentropy,
							max_epochs=200)
			
			if load_params:
				print "Loading parameters from ", param_file
				self.net.load_params_from(param_file)
			

		print "TRAINING!"
		print "input shape: ", X.shape
		print "output shape: ", y.shape
		print "Example X", X[0]
		print "Example Y", y[0]
		#print self.net.get_params()

		self.net.fit(X, y)
		print(self.net.score(X, y))


		print "Saving network parameters to ", out_param_file, "..."
		file = open(out_param_file, 'w+')
		file.close()
		self.net.save_weights_to(out_param_file)

		print "Saving filters to ", filter_file 
		self.write_filters_to_file(filter_file)

		plt = visualize.plot_loss(self.net)
		plt.show()
		plt.savefig(DIR_PROJ+'loss.png')
		plt.clf()
		plt.cla()
		
		print "Sample predictions"

		for i in range(10): 
			pred = self.net.predict(np.array([X[i]]))

			print "---------------------------------"
			print i
			print "prediction", pred
			print y[i]





'''
mini_batch_size = 1034
num_rotations=8
#num_datapoints = 4935*num_rotations
num_datapoints = 10056
#num_datapoints = 2500*num_rotations

xin, yout = makeCARRTBatches(num_datapoints, save_np_file=DIR_PROJ+'carrt_dataset')

pos = 0
neg = 0
for y in yout:
	if y == 1:
		pos += 1
	else:
		neg += 1
print '-- positive samples', pos, '| -- negative samples:', neg

param_file = DIR_PROJ+"/params1"

xin = np.load(DIR_PROJ+'carrt_dataset_x.npy')
yout = np.load(DIR_PROJ+'carrt_dataset_y.npy')

# resize and normalize input
xin = npResizeImgs(xin, X_W, X_H)
xin = normalizeInput(xin)
#xin = whitenInput(xin)


print "input size:", xin.shape, "output size:", yout.shape
#visualizeTrainingSamples(gray, names, xin, yout, rects, 8)

filter_file = DIR_PROJ+'/filters/filter'
out_param_file = DIR_PROJ+"params/param_file_carrt_i166_e200_d00"
param_file = DIR_PROJ+"pretrain/vgg_cnn/vgg_cnn_s.pkl"
graspNet = graspNet()
graspNet.train(xin, yout, param_file, out_param_file, filter_file, load_params=False, pretrain=True)
'''








'''
#xin, yout, names = makeBatches(num_datapoints, batchsize=mini_batch_size, num_rotations=num_rotations, save_np_file=DIR_PROJ+'grasp_dataset')
filter_file = DIR_PROJ+'/filters/filter'
param_file = DIR_PROJ+"/params1"
out_param_file = DIR_PROJ+"/params/params1"

xin = np.load(DIR_PROJ+'grasp_dataset_x.npy')
yout = np.load(DIR_PROJ+'grasp_dataset_y.npy')
names = np.load(DIR_PROJ+'grasp_dataset_names.npy')
gray = np.load(DIR_PROJ+'grasp_dataset_gray_images.npy')
depth = np.load(DIR_PROJ+'grasp_dataset_depth_images.npy')
rects = np.load(DIR_PROJ+'grasp_dataset_rects.npy')

# resize and normalize input
xin = npResizeImgs(xin, X_W, X_H)
xin = normalizeInput(xin)
#xin = whitenInput(xin)


print "input size:", xin.shape, "output size:", yout.shape
#visualizeTrainingSamples(gray, names, xin, yout, rects, 8)

out_param_file = DIR_PROJ+"params/param_file_10x30"
param_file = DIR_PROJ+"pretrain/vgg_cnn/vgg_cnn_s.pkl"
graspNet = graspNet()
graspNet.train(xin, yout, names, param_file, out_param_file, filter_file, load_params=False, pretrain=True)

'''

