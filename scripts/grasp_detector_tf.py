#!/usr/bin/env python
import rospy
#import grasp_net
import random
import time
import cv2
import numpy as np
from utils import *
import imutils

import dataset_reader
from config import *
import operator

from grasp_net_tf import *





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
	w_min = 16
	h_max = 60
	h_min = 8
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
		roi = rotated[r[0][1]-r[1][1]/2:r[0][1]+r[1][1]/2, r[0][0]-r[1][0]/2:r[0][0]+r[1][0]/2]
		#print r[0][1]-r[1][1]/2, r[0][1]+r[1][1]/2, r[0][0]-r[1][0]/2,r[0][0]+r[1][0]/2
		#print roi.shape
		if roi.shape[0] > 0 and roi.shape[1] > 0:

			roi = cv2.resize(roi, (X_W, X_H), interpolation=cv2.INTER_AREA)

			if len(np.array(roi).shape) > 2:
				roi = np.array(roi).reshape(roi.shape[2], roi.shape[0], roi.shape[1])
				rois.append(np.asarray(roi))
			else:
				roi = np.array([np.asarray(roi)])
				rois.append(roi)
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


tl = None
tr = None
bl = None
br = None

param_file = DIR_PROJ+"tf_grasp_model.ckpt"
graspNetTf = graspNetTf(param_file)


print "reading images"
rgb = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint159_rgb.png')
gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
depth = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint159_depth.png')
depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
normal = cv2.imread('/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/datapoint159_normal.png')
drawing = rgb.copy()


def rect_mouse(event, x, y, flags, params):
	global tl, tr, bl, br, rgb, gray, depth, normal
	global graspNetTf
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
			cv2.waitKey(-1)

			depth_rois = ROIsFromRotRects(depth, [rect], X_W, X_H)
			rgb_rois = ROIsFromRotRects(rgb, [rect], X_W, X_H)
			normal_rois = ROIsFromRotRects(normal, [rect], X_W, X_H)

			x = np.concatenate((rgb_rois, normal_rois), axis=1)
			depth_rois = np.array(depth_rois)
			depth_rois = np.reshape(depth_rois, (depth_rois.shape[0], depth_rois.shape[1], depth_rois.shape[2], 1))
			x = np.concatenate((x, depth_rois), axis=1)
			print x.shape
			x = np.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
			print x.shape
			x = normalizeInput(x)

			print 'Evaluating, with input size:', x.shape
			cv2.imshow('x', rgb_rois[0])
			results = graspNetTf.eval(x, np.array([[1,0]]))

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
	global rgb, gray, depth, normal
	global graspNetTf
	drawing = gray.copy()
	if event == cv2.EVENT_LBUTTONDOWN:
		p = [x,y]

		print "Generating rectangles"
		rects = generateRandomRectsFromPoint(drawing, p, 5000, viz=True)

		depth_rois = ROIsFromRotRects(depth, rects, X_W, X_H)
		gray_rois = ROIsFromRotRects(gray, rects, X_W, X_H)
		normal_rois = ROIsFromRotRects(normal, rects, X_W, X_H)

		gray_rois = np.array(gray_rois)
		normal_rois = np.array(normal_rois)
		depth_rois = np.array(depth_rois)
		print gray_rois.shape, normal_rois.shape, depth_rois.shape
		x = np.concatenate((gray_rois,normal_rois), axis=1)
		x = np.concatenate((x, depth_rois), axis=1)
		print x.shape
		#print "Normalizing Input"
		#x = normalizeInput(x)

		print "Example..."
		for i in range(1):
			cv2.imshow('gray', x[i*10][0].reshape(X_H, X_W,1))
			cv2.imshow('normal', x[i*10][1:4].reshape(X_H, X_W,3))
			cv2.imshow('depth', x[i*10][4:5].reshape(X_H, X_W,1))
			cv2.waitKey(30)
			print "Press any key to continue"
			cv2.waitKey(-1)

		print 'Evaluating, with input size:', x.shape
		results, correct = graspNetTf.eval(x,np.array([[1,0]]))

		indices_list = []
		score_list = []
		idx = 0
		for y in results:

			score_list.append(y[0]-y[1])
			indices_list.append(idx)
			idx += 1
		

		combined = zip(score_list, indices_list)
		combined.sort(reverse=True)
		
		num_possible = 0
		for ratio, i in combined[:10]:
			print ratio
			cv2.imshow('gray', gray_rois[i].reshape(X_H, X_W,1))
			cv2.imshow('gray', depth_rois[i].reshape(X_H, X_W,1))
			cv2.imshow('gray', normal_rois[i].reshape(X_H, X_W,3))
			drawGraspingRect(drawing, rects[i], color=(0,255,0))
			cv2.waitKey(-1)
		

		print "Done Evaluating with", num_possible, "possible grasp locations"

def getTopGraspLocations():
	global rgb, gray, depth, normal
	global graspNetTf
	drawing = gray.copy()


	X = np.zeros((0,5,X_H,X_W)) 		# neural network input
	rects_list = []						# list of rectangles describing rois to be evaluated
	print "Generating rectangles"

	# iterate over centers of rectangular grid
	for x in range((gray.shape[1]/GRID_SIZE[1]/2), gray.shape[1]-(gray.shape[1]/GRID_SIZE[1]/2), gray.shape[1]/GRID_SIZE[1]):
		for y in range((gray.shape[0]/GRID_SIZE[0]/2), gray.shape[0]-(gray.shape[0]/GRID_SIZE[0]/2), gray.shape[0]/GRID_SIZE[0]):
			print 'Center', x, y
			for p in range(NUM_SAMPLES):
				x_pos = x+random.randint(-gray.shape[0]/GRID_SIZE[0]/2, gray.shape[0]/GRID_SIZE[0]/2)
				y_pos = y+random.randint(-gray.shape[1]/GRID_SIZE[1]/2, gray.shape[1]/GRID_SIZE[1]/2)
				pos = [x_pos,y_pos]
				
				if pos[0] < 15:
					pos[0] = 15
				if pos[1] < 15:
					pos[1] = 15

				if pos[0] > gray.shape[1] - 15:
					pos[0] = gray.shape[1] - 15
				if pos[1] > gray.shape[0] - 15:
					pos[1] = gray.shape[0] - 15	

				# create the rectangles based on seed point "pos"
				print pos
				rects = generateRandomRectsFromPoint(drawing, pos, 1, viz=True)
				
				# create rois from rects
				depth_rois = ROIsFromRotRects(depth, rects, X_W, X_H)
				gray_rois = ROIsFromRotRects(gray, rects, X_W, X_H)
				normal_rois = ROIsFromRotRects(normal, rects, X_W, X_H)

				# make sure a valid roi was generated, can only generate 1 per point since ROIsFrom... not garaunteed to return rois
				while len(rects) == 0 or len(depth_rois) == 0 or len(gray_rois) == 0 or len(normal_rois) == 0:
					rects = generateRandomRectsFromPoint(drawing, pos, 1, viz=True)
					depth_rois = ROIsFromRotRects(depth, rects, X_W, X_H)
					gray_rois = ROIsFromRotRects(gray, rects, X_W, X_H)
					normal_rois = ROIsFromRotRects(normal, rects, X_W, X_H)

				rects_list.append(rects)

				# concatenate to list
				gray_rois = np.array(gray_rois)
				normal_rois = np.array(normal_rois)
				depth_rois = np.array(depth_rois)
				newx = np.concatenate((gray_rois,normal_rois), axis=1)
				newx = np.concatenate((newx, depth_rois), axis=1)
				#print "Normalizing Input"
				

				X = np.concatenate((X, newx), axis=0)

	X[:,4] = normalizeInput(X[:,4])
	X[:,0] = normalizeInput(X[:,0])
	X[:,1:3] = normalizeInput(X[:,1:3])

	# evaluate with the network
	X = normalizeInput(X)
	print 'Evaluating, with input size:', X.shape
	results, correct = graspNetTf.eval(X,np.array([[1,0]]))

	# create idices for sorting
	indices_list = []
	score_list = []
	idx = 0
	for y in results:
		score_list.append(y[0]-y[1])
		indices_list.append(idx)
		idx += 1
	
	# sort them
	combined = zip(score_list, indices_list)
	combined.sort(reverse=True)
	
	# show the best 10
	for ratio, i in combined:

		print ratio, i, len(rects_list), rects_list[i], rects_list[i][0]
		drawGraspingRect(drawing, rects_list[i][0], color=(0,255,0))
		cv2.waitKey(-1)


	print "Done Evaluating"

def probabilityMap():
	global rgb, gray, depth, normal
	global graspNetTf
	
	print gray.shape

	orig_image_size = gray.shape
	image_size = (120, 160)

	drawing = gray.copy()
	rgb = cv2.resize( rgb, image_size, interpolation=cv2.INTER_AREA)
	gray = cv2.resize( gray, image_size, interpolation=cv2.INTER_AREA)
	depth = cv2.resize( depth, image_size, interpolation=cv2.INTER_AREA)
	normal = cv2.resize( normal, image_size, interpolation=cv2.INTER_AREA)

	prob_map = np.zeros((gray.shape[0], gray.shape[1], 1))
	
	cv2.imshow('image', drawing)

	num_rects_per_pixel = 200
	# iterate over the image with padding
	for i in range(2*X_W, gray.shape[0] - 2*X_W):
		for j in range(2*X_H, gray.shape[1] - 2*X_H):
			p = [i,j]
			print p
			print "Generating rectangles..."
			
			# generate random rectangles and extract rois
			rects = generateRandomRectsFromPoint(drawing, p, num_rects_per_pixel, viz=True)
			depth_rois = ROIsFromRotRects(depth, rects, X_W, X_H)
			gray_rois = ROIsFromRotRects(gray, rects, X_W, X_H)
			normal_rois = ROIsFromRotRects(normal, rects, X_W, X_H)

			gray_rois = np.array(gray_rois)
			normal_rois = np.array(normal_rois)
			depth_rois = np.array(depth_rois)
			print gray_rois.shape, normal_rois.shape, depth_rois.shape
			x = np.concatenate((gray_rois,normal_rois), axis=1)
			x = np.concatenate((x, depth_rois), axis=1)
			#print "Normalizing Input"
			#x = normalizeInput(x)

			print "Evaluating..."
			# evaluate grasp success
			results, correct = graspNetTf.eval(x,np.array([[1,0]]))

			k = 0
			num_possible = 0
			for y in results:
				if correct[k]:
					num_possible += 1 
				k+=1

			print "Done Evaluating", i,j, "which has", num_possible, "possible grasp locations --> pixel val =", num_possible/float(num_rects_per_pixel)*255
			prob_map[i][j] = num_possible/float(num_rects_per_pixel)

			prob_map_resize = cv2.resize( prob_map, (orig_image_size[1], orig_image_size[0]), interpolation=cv2.INTER_AREA)
			cv2.imshow('probability map', prob_map_resize)
			cv2.waitKey(30)


while(cv2.waitKey(30) != 27):
	cv2.namedWindow('drawing')
	#cv2.setMouseCallback('drawing', object_mouse)
	getTopGraspLocations()
	#probabilityMap()
	cv2.imshow('drawing', rgb)