import sys, os
import numpy
import cv2
import random
import time

# iterates through all txt point clouds and creates pcds
def createPCDsFromTxt(dir):
	for file in os.listdir(dir):
		print file
		if file.endswith(".txt") and not file.contains('cneg') and not file.contains('cpos') and not file.contains('r'):
			print(file)

createPCDsFromTxt('/home/carrt/Documents/datasets/cornell_grasp/09/')