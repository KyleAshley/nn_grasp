#!/usr/bin/env python
import sys, os
import numpy as np
import cv2
import random
from utils import *
import time

IMG_TYPES = ['.jpg', '.jpeg', '.png', '.tiff']
VID_TYPES = ['.mp4', '.mov']

p1 = None
p2 = None
c3 = None
p3 = None
p4 = None

        
   

def mouseCb(event, x, y, flags, param):
	global p1, p2, c3, p3, p4
	if event == cv2.EVENT_LBUTTONDOWN:
		if p1 == None:
			p1 = [x, y]
			print 'p1', p1

		elif p1 != None and p2 == None:
			p2 = [x, y]
			print 'p2', p2

		elif p1 != None and p2 != None:
			c3 = [x, y]
			print 'c3', c3

			m = (p2[1]-p1[1])/float(p2[0]-p1[0])
			print m
			mPerp= -1.0/float(m)
			p3 = [0, 0]
			p4 = [0, 0]
			if m == 0:
				p3[0] = p2[0]
				p3[1] = c3[1]

				p4[0] = p1[0]
				p4[1] = c3[1]

			elif mPerp == 0:
				p3[0] = c3[0]
				p3[1] = p2[1]

				p4[0] = c3[1]
				p4[1] = p1[1]

			else:
				p3[0] = (c3[1]-p2[1]+mPerp*p2[0]-m*c3[0]) / (mPerp-m)
				p3[1] = p2[1]+mPerp*(p3[0]-p2[0])

				l= ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5

				if p2[1] > p1[1]:
					p4[1] = p3[1]-((m**2*l**2)/(1+m**2))**0.5
				else:
					p4[1] = p3[1]+((m**2*l**2)/(1+m**2))**0.5

				p4[0] = p3[0] + (p4[1]-p3[1])/float(m)

			print 'p3', p3
			print 'p4', p4

	
		
	


class labeller():

	def __init__(self, window):
		self.file = None 		# path to input file
		self.ftype = None		# ex: video, img, cloud etc.

		self.img = None
		self.drawing = None

		self.window = window
		self.rects = []
		self.label = None

	def start(self, file):
		print 'Type in window for Positive (p) or Negative (n) Labels?: '
		label = cv2.waitKey(-1)
		if label == 112:
			print "POSITIVE LABELS"
			self.label = 'pos'
		elif label == 110:
			print "NEGATIVE LABELS"
			self.label = 'neg'
		else:
			print "Invalid label"
			return

		self.rects = []
		global p1, p2, c3, p3, p4

		# open the file for writing
		self.file = file
		self.rect_file = self.file.strip('.png')+'_pos.txt'
		print self.rect_file
		f = open(self.rect_file, 'wa')

		print 'Reading:', file
		for t in IMG_TYPES:
			if t in file:
				self.ftype = 'IMG'
				print 'Recognized image file'
				self.img = cv2.imread(file)
				self.drawing = self.img.copy()
			else:
				for t in VID_TYPES:
					if t in file:
						self.ftype = 'VID'
						print 'Recognized video file'

		cv2.imshow(self.window, self.img)
		self.drawing = self.img.copy()

		key = cv2.waitKey(20)
		while key != 27:
			cnt = None
			# draw intermediate and final bounding boxes
			if p1 != None and p2 != None and p3 !=  None and p4 != None:
				cnt = np.array([p1, p2, p3, p4], dtype=np.int32)
				cv2.drawContours(self.drawing, [cnt], -1, (255, 255, 0), 2)
				cv2.imshow(self.window, self.drawing)
				self.drawing = self.img.copy()

				key = cv2.waitKey(-1)
				if key == 32:  	# space bar resets
					print 'clearing it'
					p1 = None
					p2 = None
					p3 = None
					p4 = None
					c3 = None

				# save it
				if key == 13 and cnt != None:
					print 'saving region:', cnt

					for c in cnt:
						p = str(c)
						p = p.strip('[')
						p = p.strip(']')
						f.write(p+'\n')
					self.rects.append(cnt)
					print 'clearing it'
					p1 = None
					p2 = None
					p3 = None
					p4 = None
					c3 = None

			cv2.drawContours(self.drawing, self.rects, -1, (0, 255, 0), 2)	
			cv2.imshow(self.window, self.drawing)	
			key = cv2.waitKey(20)
			
			





# get a path to an image
file = os.getcwd()
print file
file = file.rsplit('/', 1)[0] + '/datasets/CARRT/datapoints/datapoint14_rgb.png'
print file

window = 'Labelling Screen'
cv2.namedWindow(window)
cv2.setMouseCallback(window, mouseCb)

L = labeller(window=window)
L.start(file=file)
print L.rects

