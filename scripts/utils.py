import numpy as np
import cv2
import pickle
import pcl


# normalize and zero-center the np array and return it (shape preserving)
def normalizeInput(x):
	orig_shape = x.shape
	print 'Normalizing data...'

	new_dim = None
	for dim in x.shape:
		if new_dim == None:
			new_dim = dim
		else:
			new_dim *= dim
	x = x.reshape(new_dim)
	x = np.subtract(x, np.mean(x, axis = 0))
	x = np.divide(x, np.std(x, axis = 0))
	x = x.reshape(orig_shape)
	print 'Done!'
	return x

# TODO zero center and whitening for multimodal input (shape preserving)
def whitenInput(x):
	print 'Whitening Data...'
	orig_shape = x.shape
	x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
	x -= np.mean(x, axis = 0)			# zero center
	print 'Computing covariance matrix of data...'
	cov = np.dot(x.T, x) / x.shape[0]	# compute the covariance matrix
	print 'Performing SVD...'
	U,S,V = np.linalg.svd(cov)			# compute the SVD factorization of the data covariance matrix
	print 'Decorrelating...'
	Xrot = np.dot(x, U) 				# decorrelate the data
	Xwhite = Xrot / np.sqrt(S + 1e-5) 	# divide by the eigenvalues (which are square roots of the singular values)
	Xwhite = Xwhite.reshape(orig_shape)
	print 'Done!'
	print 'Resulting data shape:', Xwhite.shape
	return Xwhite


def npResizeImgs(arr, w, h):
	new_imgs = np.zeros((arr.shape[0], arr.shape[1], h, w), dtype=arr.dtype)
	i = 0
	for x in arr:
		j=0
		for channel in x:
			x_new = cv2.resize(channel, (w, h), interpolation=cv2.INTER_CUBIC)
			new_imgs[i][j] = x_new
			j+=1
		i+=1

	print arr.shape, new_imgs.shape
	return new_imgs



def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# http://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

# returns RGB values of encoded integer
def decode_rgb_from_pcl(rgb):
    rgb = rgb.copy()
    rgb.dtype = np.uint32
    r = np.asarray((rgb >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb & 255, dtype=np.uint8)
    rgb_arr = np.zeros((len(rgb), 3), dtype=np.uint8)
    rgb_arr[:, 2] = r
    rgb_arr[:, 1] = g
    rgb_arr[:, 0] = b
    return rgb_arr


# read PDC file into numpy array, then resaves the file as a structured pointcloud
def readPCD(fpath):
	pcdf = open(fpath)

	cloud = pcl.PointCloud()	

	stuff = pcdf.readlines()
	data = stuff[10:]

	x = np.zeros((480, 640, 1))
	y = np.zeros((480, 640, 1))
	z = np.zeros((480, 640, 1))
	rgb_img = np.zeros((480, 640, 3))
	gray_img = np.zeros((480, 640, 1))
	normal_img = np.zeros((480, 640, 3))
	xyzrgb = np.zeros((307200, 3), dtype=np.float32)
	
	rows = []
	cols = []
	for p in data:
		p.strip('\n')
		p = p.split(" ")
		idx = p[4]
		row = int(int(idx)/640)+1
		col = int(idx)%640+1
		rows.append(row)
		cols.append(col)

	row_sub = min(rows)
	col_sub = min(cols)
	for p in data:
		p.strip('\n')
		p = p.split(" ")
		idx = p[4]

		rgb_arr = decode_rgb_from_pcl(np.float32([p[3]]))
		
		if rgb_arr.any() < 0:
			print rgb_arr
		
		row = int(idx)/640 +1
		col = int(idx)%640 +1
		try:
			z[row][col][0] = float(p[2])
			y[row][col][0] = float(p[1])
			x[row][col][0] = float(p[0])
			rgb_img[row][col] = rgb_arr/255.0
			gray_img[row][col] = (rgb_img[row][col][0] + rgb_img[row][col][1] + rgb_img[row][col][2]) /(3.0)
			xyzrgb[(row*640)+col] = (float(p[0]), float(p[1]), float(p[2]))
		except:
			pass

	# make the depth image
	z = np.squeeze(z)
	z = normalized(z)*255

	#cv2.imshow('normals', model)
	#cv2.imshow('gray', gray_img)
	#cv2.imwrite(DIR_PROJ+'/pics/gray' + str(imgnum) + '.png', gray_img)
	#cv2.imwrite(DIR_PROJ + '/pics/depth' + str(imgnum) + '.png', x)
	#cv2.waitKey(-1)
	pcdf.close()
	return gray_img, z

# returns region of interest defined by a rectangle
def roi(img, rect):
	return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

# returns file len
def file_len(fname):
	i = 0
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

# shuffles numpy arrays and preserves relative ordering
def shuffle_in_unison(arr):
    rng_state = np.random.get_state()
    for l in arr:
    	np.random.shuffle(l)
    	np.random.set_state(rng_state)
    return arr


# generates a numpy array from lasagne parameter file
def createNoLearnParams(f):
	params = []

	with open(f, 'rb') as f:
		source = pickle.load(f)

	for key, values in source.items():
		if key == 'values':
			for l in np.array(values):
				params.append(l)

	return np.array(params)

# returns image rotated about the center by desired angle
def rotateImage(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def centerCrop(img, ratio=0.3):
	w = (img.shape[1]*ratio*0.3)
	h = (img.shape[0]*ratio*0.8)
	x = (img.shape[1]/2)-w/2
	y = (img.shape[0]/2)-h/2

	print 'original', img.shape[0], img.shape[1]
	print x, y, w, h
	return roi(img, (x, y, w, h))

def colorEqHist(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# convert the YUV image back to RGB format
	return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)