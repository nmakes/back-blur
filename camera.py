'''
	A simple application of opencv for Background removal / blurring

	nav.naveenvenkat@gmail.com
	github.com/nmakes/back-blur
'''

import cv2
import numpy as np
from time import time
import argparse


class StableFaceCapture:

	'''
		Notes:

		# A part of this class is taken from: github.com/nmakes/spoof/camera.py
		# This class originally contained Forehead detection and other dlib utils. We will not be using it.

		1. OpenCV considers x & y to be horizontal (left-right) and vertical (top-bottom)
		directions, while numpy considers x to be axis-0 (top-bottom) and y to be axis-1.

		2. We set a threshold to avoid unstable detection. We create a region of interest (ROI) which
		is slighly bigger than the detected face's bounding box. If face is detected again inside
		the ROI, we do not change it. If the face is not detected inside the ROI, we look for the face
		again in the whole image.

		3. We assume that only one face will be detected in the frame.
	'''

	def __init__(self, video=0, threshold=0.025, camHeight=480, camWidth=640, rescale=8, noDetectionLimit=0, cvArgs={'scaleFactor':1.1, 'minNeighbors':5, 'minSize':(30, 30),'flags':cv2.CASCADE_SCALE_IMAGE}):

		'''
			- threshold: 		this will ensure that if the captured face is within
								5% of the previously captured position, then the Fx,Fy,Fw,Fh
								will not be altered

			- noDetectionLimit:	if face is not detected for these many frames inside the ROI,
								we look for the face in the whole image

			- cvArgs={...}: 	arguments to opencv's detectMultiScale function
		'''

		# Inherit parameters
		self.threshold = threshold
		self.cvArgs = cvArgs
		self.camWidth = camWidth
		self.camHeight = camHeight
		# self.rescale = rescale

		# Initialize camera parameters
		self.cam = cv2.VideoCapture(video)
		self.cam.set(3,camWidth)
		self.cam.set(4,camHeight)
		self.camDiag = np.sqrt(self.camWidth**2 + self.camHeight**2)
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		# Region of Interest
		self.ROI = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Face capture
		self.F = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Counters for stable face detection
		self.noDetectionLimit = noDetectionLimit
		self.noDetectionCounter = 0

		# Utils for GrabCut
		# self.bgdModel = np.zeros((1,65),np.float64)
		# self.fgdModel = np.zeros((1,65),np.float64)
		# self.mm = np.zeros((int(self.camWidth/self.rescale),int(self.camHeight/self.rescale)))


	def getCamDims(self):
		# For openCV like dimensions
		return (int(self.camWidth), int(self.camHeight))


	def getCamShape(self):
		# For numpy like shape
		return (int(self.camHeight), int(self.camWidth))


	def withinThreshold(self, loc):

		dF = np.abs(self.F - np.array(loc)) / self.camDiag
		if np.all(dF <= self.threshold):
			return True
		else:
			return False


	def getCapture(self, returnSuccess=False):

		if returnSuccess==False:
			ret_val, img = self.cam.read()
			if(img.shape[:2] != (self.camHeight, self.camWidth)):
				img = cv2.resize(img, (self.camWidth, self.camHeight), cv2.INTER_CUBIC)
			return img
		else:
			ret_val, img = self.cam.read()
			if(img.shape[:2] != (self.camHeight, self.camWidth)):
				img = cv2.resize(img, (self.camWidth, self.camHeight), cv2.INTER_CUBIC)
			return ret_val, img


	def getFace(self, img=None):

		if img is None:
			img = self.getCapture()

		if self.ROI is None: # First detection attempt
			self.ROI = (0, 0, self.camWidth, self.camHeight) # (x,y,w,h)

		# Get the image inside the ROI
		roiImg = img[	int(self.ROI[1]) : int(self.ROI[1]) + int(self.ROI[3]),
						int(self.ROI[0]) : int(self.ROI[0]) + int(self.ROI[2])]

		# Get the gray image (for haar cascade classification)
		grayRoiImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Get the faces

		faces = self.faceCascade.detectMultiScale(
				grayRoiImg,
				scaleFactor=self.cvArgs['scaleFactor'],
				minNeighbors=self.cvArgs['minNeighbors'],
				minSize=self.cvArgs['minSize'],
				flags=self.cvArgs['flags']
			)

		# If no faces are found, we need to increase the noDetectionCounter, but return the previous
		# detected face region so as to avoid spikes in the estimation
		if len(faces) == 0:

			self.noDetectionCounter += 1

			if(self.noDetectionCounter>=self.noDetectionLimit):
				# self.noDetectionCounter = 0
				return None
				self.ROI = (0, 0, self.camWidth, self.camHeight)

			return self.F

		# Otherwise, reset the noDetectionCounter & continue the execution
		else:
			self.noDetectionCounter = 0

		# For the captured face(s), get the position of the face
		# Note: x, y are with respect to the image fed to the classifier. Thus,
		# these will be relative to the region of interest, and hence we add
		# ROI values to the x & y values.
		for (x,y,w,h) in faces:

			# If it is the first detection of a face, simply set the variables
			if(self.F is None):
				self.F = np.array([self.ROI[0] + x, self.ROI[1] + y, w, h])
				# self.ROI = np.array([self.F[0] - 40, self.F[1] - 40, self.F[2]+80, self.F[3] + 80])
				return self.F

			# Otherwise, check the threshold
			else:

				# If the new region is within the threshold, return the old value
				if( self.withinThreshold((x,y,w,h)) ):
					return self.F

				# Otherwise, return the new region, while setting it to be the face region
				else:
					self.F = np.array([self.ROI[0] + x, self.ROI[1] + y, w, h])
					# self.ROI = np.array([self.F[0] - 40, self.F[1] - 40, self.F[2]+80, self.F[3] + 80])
					return self.F


class BackgroundHandler:


	def __init__(self, camWidth, camHeight, rescale=8, mode='blur', alt_image_path='back.jpg', blur_kernel_size=9, mask_smooth_kernel_size=11, mask_smooth_iters=2, update_iters=10):

		self.camWidth = camWidth
		self.camHeight = camHeight
		self.rescale = rescale
		self.mode = mode
		self.alt_image_path = alt_image_path
		self.alt_image = cv2.imread(alt_image_path)
		self.alt_image = cv2.resize(self.alt_image, (self.camWidth,self.camHeight), cv2.INTER_CUBIC)

		# Counters for updating only after a few iterations
		self.update_iters = update_iters
		self.update_counter = -1

		# GrabCut runs on rescaled dimensions
		self.sW = int(self.camWidth / self.rescale)
		self.sH = int(self.camHeight / self.rescale)

		# Smoothing kernels
		self.blur_kernel_size = b = blur_kernel_size
		self.mask_smooth_kernel_size = k = mask_smooth_kernel_size
		self.mask_smooth_iters = mask_smooth_iters

		self.bgblur_kernel = np.ones((b,b), np.float32)/(b**2)
		self.mask_smooth_kernel = np.ones((k,k), np.float32)/(k**2)

		# Grabcut parameters
		self.bgdModel = np.zeros((1,65),np.float64)
		self.fgdModel = np.zeros((1,65),np.float64)
		self.mm = np.zeros((int(self.camWidth/self.rescale),int(self.camHeight/self.rescale)))
		self.mask = self.mm


	def get_mask(self, img, face_location):

		self.update_counter += 1

		x,y,w,h = face_location
		X,Y,W,H = [int(a/self.rescale) for a in (x,y,w,h)]

		if (img.shape[:2] != (self.camHeight, self.camWidth)):
			print('WARNING::BackgroundHandler::get_mask(): image shape does not match (camWidth, camHeight)')

		smallimg = cv2.resize(img, (self.sW, self.sH), interpolation=cv2.INTER_LINEAR)
		rect = (max(1,X-int(W)),max(1,Y-int(H)),min(int(3*W),self.sW),smallimg.shape[0]-(Y-int(H)))

		if (self.update_counter % self.update_iters == 0):
			self.mm, self.bgdModel, self.fgdModel = cv2.grabCut(smallimg,self.mm,rect,self.bgdModel,self.fgdModel,5,cv2.GC_INIT_WITH_RECT)
			self.mask = np.where((self.mm==2)|(self.mm==0),0,1).astype('uint8')

			self.mask = cv2.resize(self.mask, (self.camWidth,self.camHeight), interpolation=cv2.INTER_LINEAR)

			for _ in range(self.mask_smooth_iters):
				self.mask = cv2.filter2D(self.mask,-1,self.mask_smooth_kernel)

			self.mask = self.mask[:, :, np.newaxis]
		
		return self.mask


	def apply_background(self, img, mask):

		if(self.mode==None):
			return img

		elif(self.mode=='remove'):
			return img * self.mask + self.alt_image * (1-self.mask)

		elif(self.mode=='blur'):
			alt_image = cv2.resize(img, (int(self.camWidth/4), int(self.camHeight/4)))
			alt_image = cv2.filter2D(alt_image,-1,self.bgblur_kernel)
			alt_image = cv2.resize(alt_image, (self.camWidth,self.camHeight))
			return img * self.mask + alt_image * (1-self.mask)


# DEMO
if __name__=='__main__':

	mode = 'blur' # 'remove', 'blur', None
	wid, hei = (640,480)
	scale = 8
	update_iters = 10

	blur_kernel_size = 9
	mask_smooth_kernel_size = 15
	mask_smooth_iters = 5

	bg_path = 'back.jpg'

	cap = StableFaceCapture(threshold=0.1, camHeight=hei, camWidth=wid, rescale=scale)

	backhandle = BackgroundHandler(camWidth=wid, camHeight=hei, rescale=scale, 
									mode=mode, alt_image_path=bg_path, 
									blur_kernel_size=blur_kernel_size, mask_smooth_kernel_size=mask_smooth_kernel_size, 
									mask_smooth_iters=mask_smooth_iters, update_iters=update_iters)

	c = -1

	T = time()
	timer = 0
	timer_steps = 10 # To calculate FPS

	while(True):

		c+=1

		img = cap.getCapture()
		loc = cap.getFace(img)

		if loc is not None:

			mask = backhandle.get_mask(img, loc)
			img = backhandle.apply_background(img, mask)

			cv2.imshow('camera', img)

		else:
			cv2.imshow('camera', img)

		if cv2.waitKey(1) == 27:
			break  # esc to quit


		t = time()
		timer += t-T
		if(c%timer_steps==0):
			print("TIME: {} | FPS: {}".format(timer/timer_steps, timer_steps/timer))
			timer = 0
		T = t
