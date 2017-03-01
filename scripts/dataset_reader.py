import sys, os
import numpy
import cv2
import random
from utils import *
import threading
import Queue
from multiprocessing.pool import ThreadPool
import time

import config
from config import *

DIR_CORNELL = "/home/carrt/Documents/datasets/cornell_grasp/"
DIR_CARRT = "/home/carrt/Dropbox/catkin_ws/src/nn_grasp/datasets/CARRT/datapoints/"
DRAW_DATASET = False

imgnum = 0

size_div = 1					# size divider of input images
REPRESENTATIVE_IMAGE = 4		# image from each interaction to use

# Parameters:
# datapoints: number of total labelled rectangles in the folder
# flip: when set, creates a copy of each datapoint that is flipped along the vertical axis
# onehot: when set, generates one-hot encoded output for binary classification [1 0] == True, [0 1] == False
# grayscale: when set, converts RGB images to grayscale
def loadDataCARRTClassification(datapoints, flip=True, onehot=True, verbose=False, grayscale=True):

	if flip:
		if grayscale:
			x = np.zeros((datapoints*2, 5, X_H, X_W), dtype=float)
		else:
			x = np.zeros((datapoints*2, 7, X_H, X_W), dtype=float)
		if onehot:
			y = np.zeros((datapoints*2, 2), dtype=float)
		else:
			y = np.zeros((datapoints*2, 1), dtype=float)
	else:
		if grayscale:
			x = np.zeros((datapoints, 5, X_H, X_W), dtype=float)
		else:
			x = np.zeros((datapoints, 7, X_H, X_W), dtype=float)
		if onehot:
			y = np.zeros((datapoints, 2), dtype=float)
		else:
			y = np.zeros((datapoints, 1), dtype=float)

	names = []
	depth_list = []
	rgb_list = []
	normals_list = []
	all_rects = None
	i=0
	DIR = DIR_CARRT

	num_datapoints = 0

	for interaction in range(1,245):
		t1 = time.time()
		print "Grasp number:", interaction

		rgb = cv2.imread(DIR+'datapoint'+str(interaction)+'_rgb.png')
		depth = cv2.imread(DIR+'datapoint'+str(interaction)+'_depth.png')
		normal = cv2.imread(DIR+'datapoint'+str(interaction)+'_normal.png')

		pos_path = DIR+'datapoint'+str(interaction)+"_rgcpos.txt"
		neg_path = DIR+'datapoint'+str(interaction)+"_rgcneg.txt"
		posf = open(pos_path)
		negf = open(neg_path)

		# read in rects
		pos_rects = []
		neg_rects = []
		flen_pos = file_len(pos_path)
		flen_neg = file_len(neg_path)

		if verbose:
			print "	--has", flen_pos/4, "positives"
			print "	--has", flen_neg/4, "negatives"
		num_datapoints+= flen_pos/4
		num_datapoints+= flen_neg/4

		
		# create list of corner points for pos/neg rects
		for rr in range(flen_pos/4):
			pos_rects.append([])
			for point in range(4):
				xx,yy = posf.readline().split()
				pos_rects[rr].append((float(xx),float(yy)))

		for rr in range(flen_neg/4):
			neg_rects.append([])
			for point in range(4):
				xx,yy = negf.readline().split()
				neg_rects[rr].append((float(xx),float(yy)))

		pos_rects = np.nan_to_num(pos_rects)
		neg_rects = np.nan_to_num(neg_rects)

		#np.random.shuffle(pos_rects)
		#np.random.shuffle(neg_rects)
		# make datasets even by shaving off some negative samples
		#neg_rects = neg_rects[:len(pos_rects)+1000]
		
		pos_neg_rects = None

		if all_rects == None:
			all_rects = np.concatenate((pos_rects, neg_rects), axis=0)
			pos_neg_rects = all_rects.copy()
		elif len(pos_rects) > 0 and len(neg_rects) > 0:
			#print "Previous rects:", all_rects.shape
			all_rects = np.concatenate((np.concatenate((pos_rects, neg_rects), axis=0), all_rects), axis=0)
			pos_neg_rects = np.concatenate((pos_rects, neg_rects), axis=0)
		else:
			#print "Previous rects:", all_rects.shape
			if len(pos_rects) > 0:
				all_rects = np.concatenate((pos_rects, all_rects), axis=0)
				pos_neg_rects = pos_rects.copy()
			else:
				all_rects = np.concatenate((neg_rects, all_rects), axis=0)
				pos_neg_rects = neg_rects.copy()

		if verbose:
			print "Concatenating:", pos_rects.shape, neg_rects.shape, "to:", all_rects.shape
			print "total rectangles:", len(pos_neg_rects)
			print pos_neg_rects

		to_del = [] 	# list of indices to delete
		for rrr in range(len(pos_neg_rects)):

			rect_points = np.array(np.int0(pos_neg_rects[rrr]))
			rectangle = cv2.minAreaRect(rect_points)
			angle = rectangle[2]
			rect_w = int(rectangle[1][0])
			rect_h = int(rectangle[1][1])
			pos_x = int(rectangle[0][0])
			pos_y = int(rectangle[0][1])

			if rect_w > 400 or rect_h > 400 or rect_h == 0 or rect_w == 0:
				print "Rectangle size too large or small, ignoring..."
				to_del.append(i)

			else:
				box = cv2.boxPoints(rectangle)
				box = np.int0(box)

				# positive sample output == 1
				if rrr < pos_rects.shape[0]:
					if onehot:
						y[i] = [1,0]
					else:
						y[i] = 1

					if flip:
						if onehot:
							y[i+1] = [1,0]
						else:
							y[i+1] = 1

					if verbose:
						print "Positive datapoint", i 
						print rectangle
				else:
					if onehot:
						y[i] = [0,1]
					else:
						y[i] = 0

					if flip:
						if onehot:
							y[i+1] = [0,1]
						else:
							y[i+1] = 0
					if verbose:
						print "Negative datapoint", i 
						print rectangle
				
				if verbose:	
					print "Angle:", rectangle[2]
				# adjust for weird angle convention
				if rectangle[2] < -45.0:
					angle += 90.0
					temp = rect_w
					rect_w = rect_h
					rect_h = temp
					if verbose:
						print "Rotating ROI +90"

				# i dont even know... dont mess with it
				if rect_w < rect_h:
					angle += 90
					temp = rect_w
					rect_w = rect_h
					rect_h = temp

				if verbose:
					print rect_w, rect_h
					print "y["+str(i)+"]", y[i], "y["+str(i+1)+"]", y[i+1]
					#rect_w = rect_w + 4
				
				if DRAW_DATASET:
					drawing = rgb.copy()

				depth_rot = rotateImage(depth, angle, (pos_x, pos_y))
				rgb_rot = rotateImage(rgb, angle, (pos_x, pos_y))
				normal_rot = rotateImage(normal, angle, (pos_x, pos_y))

				if DRAW_DATASET:
					drawing_rot = rgb_rot.copy()
				
				depth_roi = depth_rot[pos_y-rect_h/2:pos_y+rect_h/2, pos_x-rect_w/2:pos_x+rect_w/2]
				rgb_roi = rgb_rot[pos_y-rect_h/2:pos_y+rect_h/2, pos_x-rect_w/2:pos_x+rect_w/2]
				normal_roi = normal_rot[pos_y-rect_h/2:pos_y+rect_h/2, pos_x-rect_w/2:pos_x+rect_w/2]

				#print rgb_roi.shape, depth_roi.shape, normal_roi.shape


				if rect_w == 0 or rect_h == 0:
					depth_roi_resize = depth_roi
					rgb_roi_resize = rgb_roi
					normal_roi_resize = normal_roi

				elif rect_w < rect_h:
					depth_roi = rotateImage(depth_roi, -90, (rect_w/2, rect_h/2))
					rgb_roi = rotateImage(rgb_roi, -90, (rect_w/2, rect_h/2))
					normal_roi = rotateImage(normal_roi, -90, (rect_w/2, rect_h/2))

				if DRAW_DATASET:
					cv2.imshow('rgb', rgb_roi)

				try:
					depth_roi_resize = cv2.resize(depth_roi, (X_W, X_H), interpolation=cv2.INTER_CUBIC)
					rgb_roi_resize = cv2.resize(rgb_roi, (X_W, X_H), interpolation=cv2.INTER_CUBIC)
					normal_roi_resize = cv2.resize(normal_roi, (X_W, X_H), interpolation=cv2.INTER_CUBIC)
				except:
					print "Error resizing image... perhaps this is bad...perhaps not"

				if DRAW_DATASET:
					cv2.imshow('rgb resized', rgb_roi_resize)

				depth_roi_no_scale = depth_roi.copy()
				rgb_roi_no_scale = rgb_roi.copy()  
				normal_roi_no_scale = normal_roi.copy()

				depth_roi_resize2 = cv2.cvtColor(depth_roi_resize,cv2.COLOR_BGR2GRAY)

				# histogram EQ
				#rgb_roi_resize = colorEqHist(rgb_roi_resize)
				#normal_roi_resize = colorEqHist(normal_roi_resize)
				#depth_roi_resize2 = cv2.equalizeHist(depth_roi_resize2)
				
				if grayscale:
					gray_roi_resize = cv2.cvtColor(rgb_roi_resize,cv2.COLOR_BGR2GRAY)
					x[i][0:1] = gray_roi_resize.reshape((1, X_H, X_W))
					x[i][1:4] = normal_roi_resize.reshape((3, X_H, X_W))
					x[i][4:5] = depth_roi_resize2.reshape((1, X_H, X_W))

					if flip:
						gray_flip = cv2.flip(gray_roi_resize, 1)
						normal_flip = cv2.flip(normal_roi_resize, 1)
						depth_flip = cv2.flip(depth_roi_resize2, 1)

						x[i+1][0:1] = gray_flip.reshape((1, X_H, X_W))
						x[i+1][1:4] = normal_flip.reshape((3, X_H, X_W))
						x[i+1][4:5] = depth_flip.reshape((1, X_H, X_W))

				else:
					x[i][0:3] = rgb_roi_resize.reshape((3, X_H, X_W))
					x[i][3:6] = normal_roi_resize.reshape((3, X_H, X_W))
					x[i][6:7] = depth_roi_resize2.reshape((1, X_H, X_W))

					if flip:
						rgb_flip = cv2.flip(rgb_roi_resize, 1)
						normal_flip = cv2.flip(normal_roi_resize, 1)
						depth_flip = cv2.flip(depth_roi_resize2, 1)

						x[i+1][0:3] = rgb_flip.reshape((3, X_H, X_W))
						x[i+1][3:6] = normal_flip.reshape((3, X_H, X_W))
						x[i+1][6:7] = depth_flip.reshape((1, X_H, X_W))

				if DRAW_DATASET:
					cv2.drawContours(drawing,[box],0,(0,0,255),2)
					#cv2.rectangle(drawing, (CROI_X, CROI_Y), ((CROI_X+CROI_W), (CROI_Y+CROI_H)),(0,0,255),2)
					cv2.circle(drawing_rot, (pos_x-rect_w/2, pos_y-rect_h/2), 1, (255, 255, 255), 2)
					cv2.circle(drawing_rot, (pos_x-rect_w/2, pos_y+rect_h/2), 1, (255, 255, 255), 2)
					cv2.circle(drawing_rot, (pos_x+rect_w/2, pos_y-rect_h/2), 1, (255, 255, 255), 2)
					cv2.circle(drawing_rot, (pos_x+rect_w/2, pos_y+rect_h/2), 1, (255, 255, 255), 2)

					cv2.imshow('depth_roi', depth_roi_resize2)
					cv2.imshow('rgb_roi', rgb_roi_resize)
					cv2.imshow('normal_roi', normal_roi_resize)

					#cv2.imshow('FLIP_depth_roi', depth_flip)
					#cv2.imshow('FLIP_rgb_roi', rgb_flip)
					#cv2.imshow('FLIP_normal_roi', normal_flip)

					cv2.imshow('drawing', drawing)
					cv2.imshow('drawing_rot', drawing_rot)
					cv2.waitKey(-1)

				if verbose:
					print "\tStoring as datapoint", i
					print "\tInput shape:", x[i].shape, "Output Shape:", y[i].shape

				if flip:
					i+=2
				else:
					i+=1


	print "Total datapoints", i

	return x, y


def loadDataCornellClassification(graspnum, datapoint, batchsize, verbose=True, num_rotations=10):

	x = np.zeros((batchsize*8, 3, X_H, X_W), dtype=float)
	y = np.zeros((batchsize*8, 1), dtype=float)
	names = []
	depth_list = []
	img_list = []
	all_rects = None
	i=0
	DIR = DIR_CORNELL

	# change to the path for the first time
	if os.path.isdir(DIR+"0"+str(graspnum/100)):
		print "Changing paths", DIR+"0"+str(graspnum/100)
		os.chdir(DIR+"0"+str(graspnum/100))
	
	for interaction in range(graspnum, graspnum+batchsize):
		t1 = time.time()
		print "Grasp number:", graspnum

		# generate file names
		if interaction < 100:
			img_path = DIR+"0"+str(graspnum/100)+ "/pcd00"+ str(interaction)+'r.png'
			pos_path = DIR+"0"+str(graspnum/100)+ "/pcd00"+ str(interaction)+'cpos.txt'
			neg_path = DIR+"0"+str(graspnum/100)+ "/pcd00"+ str(interaction)+'cneg.txt'
			pcd_path = DIR+"0"+str(graspnum/100)+ "/pcd00"+ str(interaction)+'.txt'
		else:
			img_path = DIR+"0"+str(graspnum/100)+ "/pcd0"+ str(interaction)+'r.png'
			pos_path = DIR+"0"+str(graspnum/100)+ "/pcd0"+ str(interaction)+'cpos.txt'
			neg_path = DIR+"0"+str(graspnum/100)+ "/pcd0"+ str(interaction)+'cneg.txt'
			pcd_path = DIR+"0"+str(graspnum/100)+ "/pcd0"+ str(interaction)+'.txt'

		if verbose:
			print "----------------------------------------------"
			print i, img_path

		# open up the image files
		if os.path.exists(pcd_path):
			img, depth_img = readPCD(pcd_path)
			#print "input shapes"
			#print img.shape, depth_img.shape
			img_list.append(img)
			depth_list.append(depth_img)
		else:
			print "Failed to read pcd file...skipping"

		# check to make sure we got good image data
		if img is not None and depth_img is not None and os.path.exists(pos_path) and os.path.exists(neg_path):

			if verbose:
				pass
				#print img.shape
				#print depth_img.shape
			if DRAW_DATASET:
				drawing = img.copy()

			posf = open(pos_path)
			negf = open(neg_path)

			# read in rects
			pos_rects = []
			neg_rects = []
			flen_pos = file_len(pos_path)
			flen_neg = file_len(neg_path)

			if verbose:
				print pos_path, "has", flen_pos/4, datapoint
				print neg_path, "has", flen_neg/4, datapoint

			# create list of corner points for pos/neg rects
			for rr in range(flen_pos/4):
				pos_rects.append([])
				for point in range(4):
					xx,yy = posf.readline().split()
					pos_rects[rr].append((float(xx),float(yy)))

			for rr in range(flen_neg/4):
				neg_rects.append([])
				for point in range(4):
					xx,yy = negf.readline().split()
					neg_rects[rr].append((float(xx),float(yy)))

			pos_rects = np.nan_to_num(pos_rects)
			neg_rects = np.nan_to_num(neg_rects)
			pos_neg_rects = None

			if all_rects == None:
				all_rects = np.concatenate((pos_rects, neg_rects), axis=0)
				pos_neg_rects = all_rects.copy()
			elif len(pos_rects) > 0 and len(neg_rects) > 0:
				print "Previous rects:", all_rects.shape
				all_rects = np.concatenate((np.concatenate((pos_rects, neg_rects), axis=0), all_rects), axis=0)
				pos_neg_rects = np.concatenate((pos_rects, neg_rects), axis=0)
			else:
				print "Previous rects:", all_rects.shape
				if len(pos_rects) > 0:
					all_rects = np.concatenate((pos_rects, all_rects), axis=0)
					pos_neg_rects = pos_rects.copy()
				else:
					all_rects = np.concatenate((neg_rects, all_rects), axis=0)
					pos_neg_rects = neg_rects.copy()
			print "Concatenating:", pos_rects.shape, neg_rects.shape, "to:", all_rects.shape

			to_del = [] 	# list of indices to delete
			for rrr in range(len(pos_neg_rects)):

				rect_points = np.array(np.int0(pos_neg_rects[rrr]))
				rectangle = cv2.minAreaRect(rect_points)
				angle = rectangle[2]
				rect_w = int(rectangle[1][0])
				rect_h = int(rectangle[1][1])
				pos_x = int(rectangle[0][0])
				pos_y = int(rectangle[0][1])

				if rect_w > 400 or rect_h > 400 or rect_h == 0 or rect_w == 0:
					print "Rectangle size too large or small, ignoring..."
					to_del.append(i)

				else:
					names.append([img_path])
					box = cv2.boxPoints(rectangle)
					box = np.int0(box)

					# positive sample output == 1
					if rrr < pos_rects.shape[0]:
						y[i] = 1
						if verbose:
							print "Positive datapoint", i 
							print rectangle
					else:
						if verbose:
							print "Negative datapoint", i 
							print rectangle
						y[i] = 0
					
					# adjust for weird angle convention
					if rectangle[2] < -45.0:
						angle += 90.0;
						temp = rect_w
						rect_w = rect_h
						rect_h = temp
						print "Rotating ROI +90"

					print rect_w, rect_h
					#rect_w = rect_w + 4
					
					if DRAW_DATASET:
						drawing = img.copy()

					depth_rot = rotateImage(depth_img, angle, (pos_x, pos_y))
					gray_rot = rotateImage(img, angle, (pos_x, pos_y))

					if DRAW_DATASET:
						drawing_rot = gray_rot.copy()
					
					depth_roi = depth_rot[pos_y-rect_h/2.0:pos_y+rect_h/2.0, pos_x-rect_w/2.0:pos_x+rect_w/2.0]
					gray_roi = gray_rot[pos_y-rect_h/2.0:pos_y+rect_h/2.0, pos_x-rect_w/2.0:pos_x+rect_w/2.0]

					try:
						if rect_w == 0 or rect_h == 0:
							depth_roi_resize = depth_roi
							gray_roi_resize = gray_roi

						elif rect_w < rect_h:
							depth_roi = rotateImage(depth_roi, -90, (rect_w/2, rect_h/2))
							gray_roi = rotateImage(gray_roi, -90, (rect_w/2, rect_h/2))

						else:

							depth_roi_no_scale = depth_roi.copy()
							gray_roi_no_scale = gray_roi.copy()

							#depth_roi_resize = np.zeros((100, 50))
							#gray_roi_resize = np.zeros((100, 50))
							#fx = 100/depth_roi.shape[1]
							#fy = 50/depth_roi.shape[0]
							depth_roi_resize = cv2.resize(depth_roi, (X_W, X_H), interpolation=cv2.INTER_CUBIC)
							gray_roi_resize = cv2.resize(gray_roi, (X_W, X_H), interpolation=cv2.INTER_CUBIC)

						print gray_roi_resize.shape, gray_roi.shape

					except ValueError:
						break

					x[i][0] = depth_roi_resize
					x[i][1] = gray_roi_resize
					x[i][2] = depth_roi_resize

					if DRAW_DATASET:
						cv2.drawContours(drawing,[box],0,(0,0,255),2)
						#cv2.rectangle(drawing, (CROI_X, CROI_Y), ((CROI_X+CROI_W), (CROI_Y+CROI_H)),(0,0,255),2)
						cv2.circle(drawing_rot, (pos_x-rect_w/2, pos_y-rect_h/2), 1, (255, 255, 255), 2)
						cv2.circle(drawing_rot, (pos_x-rect_w/2, pos_y+rect_h/2), 1, (255, 255, 255), 2)
						cv2.circle(drawing_rot, (pos_x+rect_w/2, pos_y-rect_h/2), 1, (255, 255, 255), 2)
						cv2.circle(drawing_rot, (pos_x+rect_w/2, pos_y+rect_h/2), 1, (255, 255, 255), 2)

						cv2.imshow('depth_roi', depth_roi_resize)
						cv2.imshow('gray_roi', gray_roi_resize)
						cv2.imshow('depth_roi_no_scale', depth_roi_no_scale)
						cv2.imshow('gray_roi_no_scale', gray_roi_no_scale)
						cv2.imshow('drawing', drawing)
						cv2.imshow('drawing_rot', drawing_rot)
						cv2.waitKey(-1)

					if verbose:
						print "\tStoring as datapoint", i
						print "\tInput shape:", x[i].shape, "Output Shape:", y[i].shape
	 
					i+=1
					datapoint+=1

			for ii in to_del:
				all_rects = np.delete(all_rects, (ii), axis=0)

			print i, all_rects.shape[0]
			if i != all_rects.shape[0]:
				print "ERROR"
				return

		t2 = time.time()
		print "time: " + str(t2-t1)

		graspnum+=1
		datapoint=0
		interaction+=1


	# return batches with filenames and current graspnum and datapoint in that file where we left off
	return x[:i], y[:i], names[:i], depth_list, img_list, all_rects


def makeCARRTBatches(num_datapoints=100, save_np_file=None, verbose=False):
	x, y = loadDataCARRTClassification(datapoints=num_datapoints, verbose=verbose)
	
	x = np.asarray(x, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)

	print "DONE!"
	print "Batch dimensions"
	print 'x', x.shape, 'y', y.shape

	arr = [x, y]
	print "Randomizing data..."
	arr = shuffle_in_unison(arr)

	if save_np_file:
		print "saving dataset to: ", save_np_file
		np.save(save_np_file+'_x', arr[0])
		np.save(save_np_file+'_y', arr[1])

	return x, y


def makeCornellBatches(num_datapoints, batchsize = 50, verbose=True, lr='left', num_rotations=5, save_np_file=None):
	
	x = []
	y = []
	names = []
	depth_list = []
	img_list = []
	rects_list = []
	graspnum = 100	# starting grasp number
	datapoint = 0	# offset

	xx, yy, nn, depth_list, img_list, rects_list = loadDataCornellClassification(graspnum, datapoint, num_datapoints, verbose=verbose)
	x = xx
	y = yy
	names = nn

	x = np.asarray(x, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)
	names = np.asarray(names)
	depth_np = np.asarray(depth_list)
	img_np = np.asarray(img_list)
	rects_np = np.asarray(rects_list)

	print "DONE!"
	print "Batch dimensions"
	print 'x', x.shape, 'y', y.shape, 'n', names.shape, 'r', rects_list.shape

	arr = [x, y, names, depth_np, img_np, rects_np]
	if verbose:
		print "Randomizing data..."
	arr = shuffle_in_unison(arr)

	if save_np_file:
		print "saving dataset to: ", save_np_file
		np.save(save_np_file+'_x', arr[0])
		np.save(save_np_file+'_y', arr[1])
		np.save(save_np_file+'_names', arr[2])
		np.save(save_np_file+'_depth_images', arr[3])
		np.save(save_np_file+'_gray_images', arr[4])
		np.save(save_np_file+'_rects', arr[5])

	return x, y, names

