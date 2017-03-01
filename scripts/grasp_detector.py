#!/usr/bin/env python
import rospy
#import grasp_net
import random
import time
import cv2
import numpy as np
from utils import *
import imutils
from grasp_net import *
import dataset_reader
import config
import operator

DIR_PROJ = '/home/carrt/Dropbox/catkin_ws/src/nn_grasp/'




# segments an image into a grid for higher resolution grasp detection as per: http://pjreddie.com/media/files/papers/grasp_detection.pdf
def getImageGrid(img, depth_img, grid_size):
	w = img.shape[0]
	h = img.shape[1]

	# 2d list of images containing subdivided regions of original image
	rgb_grid = []
	depth_grid = []

	for i in range(grid_size[0]):
		rgb_row = []
		depth_row = []

		for j in range(grid_size[1]):
			grid_w = w/grid_size[0]
			grid_h = h/grid_size[1]
			center_x = grid_w * i + grid_w/2
			center_y = grid_h * j + grid_h/2

			# extract roi of grid area
			sub_img = img[center_y-grid_h/2.0:center_y+grid_h/2.0, center_x-grid_w/2.0:center_x+grid_w/2.0]
			sub_depth_img = depth_img[center_y-grid_h/2.0:center_y+grid_h/2.0, center_x-grid_w/2.0:center_x+grid_w/2.0]

			rgb_row.append(sub_img)
			depth_row.append(sub_depth_img)

		rgb_grid.append(rgb_row)
		depth_grid.append(depth_row)


# generate a set of random rects near a single point in the image
def generateRandomRectsFromPoint(img, point, num, viz=False):
	w = img.shape[1]
	h = img.shape[0]
	rects = []

	# amount of randomness in x/y positions of random patches
	x_variance = 20
	y_variance = 20
	w_max = 120
	w_min = 30
	h_max = 60
	h_min = 10
	for i in range(num):
		random.seed(time.time())
		x = random.randint(max(point[0] - x_variance/2, 0), point[0]+x_variance/2)
		y = random.randint(max(point[1] - y_variance/2, 0), point[1]+y_variance/2)
		w = random.randint(w_min, w_max)
		h = random.randint(h_min, h_max)
		theta = random.randint(0, 360)
		rect = ((x, y), (w, h), theta)
		rects.append(rect)
	return rects


# generates a set of random rectangles for consideration when grasping on the entire image
def generateRandomRects(img, grid_size, num_per_cell, viz=False):
	w = img.shape[1]
	h = img.shape[0]
	print img.shape
	rects = []

	grid_w = w/grid_size[1]
	grid_h = h/grid_size[0]

	# iterate over the grid
	for i in range(grid_size[0]):
		for j in range(grid_size[1]):

			center_x = grid_w * i + grid_w/2
			center_y = grid_h * j + grid_h/2

			if RANDOM_RECTS:
				for r in range(num_per_cell):
					# create randomized rectangles
					random.seed(time.time())
					pos_x = random.randint(max(center_x - grid_w/2, 0), center_x+grid_w/2)
					pos_y = random.randint(max(center_y - grid_h/2, 0), center_y+grid_h/2)
					theta = random.randint(0, 360)
					w = random.randint(grid_w/6, grid_w + grid_w/2)
					h = random.randint(grid_h/6, grid_h + grid_h/2)

					rect = ((pos_x, pos_y), (w, h), theta)
					rects.append(rect)
			else:
				for r_w in range(center_x - grid_w/2, center_x + grid_w/2, 1):
					for r_h in range(center_y - grid_h/2, center_y + grid_h/2, 1):
						# create randomized rectangles
						random.seed(time.time())
						pos_x = r_w
						pos_y = r_h
						print pos_x, pos_y
						for t in range(0, 360, 4):
							theta = t
							w = random.randint(grid_w/6, grid_w + grid_w/2)
							h = random.randint(grid_h/6, grid_h + grid_h/2)

							rect = ((pos_x, pos_y), (w, h), theta)
							rects.append(rect)
	'''
	if viz:
		drawing = img.copy()
		for r in rects:
			box = cv2.boxPoints(r)
			box = np.int0(box)
			cv2.drawContours(drawing,[box],0,(0,0,255),2)
		cv2.imshow('random rects', drawing)
		cv2.waitKey(-1)
	'''
	return rects

def drawGraspingRect(img, r, color=(0,0,255)):
	drawing = img.copy()
	box = cv2.boxPoints(r)
	box = np.int0(box)
	cv2.drawContours(drawing,[box],0,(0,0,255),2)
	cv2.imshow('Rotated Rect', drawing)
	cv2.waitKey(-1)

def ROIsFromRotRects(img, rects, X_W, X_H):
	rois = []
	for r in rects:
		rotated = rotateImage(img, r[2], (r[0][0], r[0][1]))
		roi = rotated[r[0][1]-r[1][1]/2.0:r[0][1]+r[1][1]/2.0, r[0][0]-r[1][0]/2.0:r[0][0]+r[1][0]/2.0]
		
		if roi.shape[0] > 0 and roi.shape[1] > 0:
			#print roi.shape
			roi = cv2.resize(roi, (X_W, X_H), interpolation=cv2.INTER_AREA)
			rois.append(np.asarray(roi))
			'''
			
			cv2.imshow('roi',roi)
			drawGraspingRect(img, r)
			cv2.waitKey(-1)
			'''
			pass
		
	return rois





#--------------------------------------------------------------------------------------------------------------#
# Grasp detection on a single image            

#gray, depth = readPCD('/home/carrt/Documents/datasets/cornell_grasp/06/pcd0602.txt')
#gray = centerCrop(gray)
#depth = centerCrop(depth)

if not LIVE_DRAW:
	rgb = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint1_rgb.png')
	depth = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint1_depth.png')
	depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
	normal = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint1_normal.png')

	drawing = rgb.copy()

	rects = generateRandomRects(rgb, GRID_SIZE, NUM_SAMPLES, viz=False)
	depth_rois = ROIsFromRotRects(depth, rects, X_W, X_H)
	rgb_rois = ROIsFromRotRects(rgb, rects, X_W, X_H)
	normal_rois = ROIsFromRotRects(normal, rects, X_W, X_H)

	param_file = DIR_PROJ+"params/param_file_carrt_i166_e100"
	graspNet = graspNet(param_file)

	x = np.concatenate((rgb_rois, normal_rois), axis=3)
	depth_rois = np.array(depth_rois)
	depth_rois = np.reshape(depth_rois, (depth_rois.shape[0], depth_rois.shape[1], depth_rois.shape[2], 1))
	x = np.concatenate((x, depth_rois), axis=3)
	print x.shape
	x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
	print x.shape
	x = x.astype(theano.config.floatX)
	x = normalizeInput(x)



	print 'Evaluating, with input size:', x.shape
	results = graspNet.eval(x).flatten()


	img = np.zeros(rgb.shape)
	print img.shape
	for r in range(len(results)):
		try:
			x = rects[r][0][1]
			y = rects[r][0][0]
			img[x][y] += results[r]
		except:
			pass

	cv2.imshow('img', img*2)
	cv2.waitKey(-1)


	sorted_results = sorted(range(len(results)), reverse=False, key=lambda x: results[x])[-10:]

	for i in range(len(results)):
		print "confidence:", results[i]
		drawGraspingRect(drawing, rects[i], color=(0,255,0))
		cv2.imshow('img', drawing)
		#cv2.imshow('roi', np.reshape(x[i], x[i])
		cv2.waitKey(-1)


	'''
	filter_file = DIR_PROJ+'/filters/filter'
	param_file = DIR_PROJ+"/params_depth"
	out_param_file = DIR_PROJ+"/params/params1"

	graspNet = graspNet()
	graspNet.train(xin, yout, names, param_file, out_param_file, filter_file, load_params=False, pretrain=False)
	'''

else:
	tl = None
	tr = None
	bl = None
	br = None

	param_file = DIR_PROJ+"params/param_file_carrt_i166_e100"
	graspNet = graspNet(param_file)

	
	rgb = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint1_rgb.png')
	depth = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint1_depth.png')
	depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
	normal = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint1_normal.png')
	drawing = rgb.copy()

	def rect_mouse(event, x, y, flags, params):
		global tl, tr, bl, br, rgb, depth, normal
		global graspNet
		if event == cv2.EVENT_LBUTTONDOWN:
			if tl == None:
				tl = [x, y]
			elif tr == None:
				tr = [x,y]
			elif bl == None:
				bl = [x,y]
			elif br == None:
				br = [x,y]
			elif tl != None and tr != None and bl != None and br != None:
				print tl, tr, bl, br
				rect = cv2.minAreaRect(np.int0([tl, tr, bl, br]))
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(drawing,[box],0,(0,0,255),2)
				cv2.imshow('drawing', drawing)

				depth_rois = ROIsFromRotRects(depth, [rect], X_W, X_H)
				rgb_rois = ROIsFromRotRects(rgb, [rect], X_W, X_H)
				normal_rois = ROIsFromRotRects(normal, [rect], X_W, X_H)

				x = np.concatenate((rgb_rois, normal_rois), axis=3)
				depth_rois = np.array(depth_rois)
				depth_rois = np.reshape(depth_rois, (depth_rois.shape[0], depth_rois.shape[1], depth_rois.shape[2], 1))
				x = np.concatenate((x, depth_rois), axis=3)
				print x.shape
				x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
				print x.shape
				x = x.astype(theano.config.floatX)
				x = normalizeInput(x)

				print 'Evaluating, with input size:', x.shape
				results = graspNet.eval(x).flatten()

				for i in range(len(results)):
					print "confidence:", results[i]

				tl = None
				tr = None
				bl = None
				br = None

			else:
				tl = None
				tr = None
				bl = None
				br = None

		else:
			pass

	def object_mouse(event, x, y, flags, params):
		global rgb, depth, normal
		global graspNet
		drawing = rgb.copy()
		if event == cv2.EVENT_LBUTTONDOWN:
			p = [x,y]
			rects = generateRandomRectsFromPoint(drawing, p, 8000, viz=True)

			depth_rois = ROIsFromRotRects(depth, rects, X_W, X_H)
			rgb_rois = ROIsFromRotRects(rgb, rects, X_W, X_H)
			normal_rois = ROIsFromRotRects(normal, rects, X_W, X_H)

			x = np.concatenate((rgb_rois, normal_rois), axis=3)
			depth_rois = np.array(depth_rois)
			depth_rois = np.reshape(depth_rois, (depth_rois.shape[0], depth_rois.shape[1], depth_rois.shape[2], 1))
			x = np.concatenate((x, depth_rois), axis=3)
			print x.shape
			x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
			print x.shape
			x = x.astype(theano.config.floatX)
			x = normalizeInput(x)

			print 'Evaluating, with input size:', x.shape
			results = graspNet.eval(x).flatten()

			results = dict(zip(results, range(len(results))))
			sorted_results = sorted(results.items(), reverse=False, key=operator.itemgetter(0))[-10]
			for key, val in sorted_results:
				print key, val
				print "confidence:", key
				drawGraspingRect(drawing, rects[val], color=(0,255,0))
				cv2.imshow('img', drawing)
				#cv2.imshow('roi', np.reshape(x[i], x[i])
				cv2.waitKey(-1)

	while(cv2.waitKey(30) != 27):
		cv2.namedWindow('drawing')
		cv2.setMouseCallback('drawing', object_mouse)
		cv2.imshow('drawing', rgb)