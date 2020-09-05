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

		# Initialize camera and cascade classifier
		self.cam = cv2.VideoCapture(video)
		self.cam.set(3,camWidth)
		self.cam.set(4,camHeight)
		# self.detector=dlib.get_frontal_face_detector()
		self.camWidth = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.camHeight = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.camDiag = np.sqrt(self.camWidth**2 + self.camHeight**2)
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		# self.faceCascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

		# Region of Interest
		self.ROI = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Face capture
		self.F = None # np.array([0,0,0,0]) # (x,y,w,h)

		# Counters for stable face detection
		self.noDetectionLimit = noDetectionLimit
		self.noDetectionCounter = 0

		# Utils for GrabCut
		self.bgdModel = np.zeros((1,65),np.float64)
		self.fgdModel = np.zeros((1,65),np.float64)
		self.mm = np.zeros((int(self.camWidth/rescale),int(self.camHeight/rescale)))


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
			return img
		else:
			ret_val, img = self.cam.read()
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



# DEMO
if __name__=='__main__':


	mode = 'remove' # 'remove', 'blur', None


	wid, hei = (640,480)
	scale = 8
	sW, sH = (int(wid/scale), int(hei/scale))

	cap = StableFaceCapture(threshold=0.1, camHeight=hei, camWidth=wid, rescale=scale)

	c = -1

	bgimg = cv2.imread('back.jpg')
	bgimg = cv2.resize(bgimg, (wid,hei), cv2.INTER_CUBIC)

	mean_kernel_size = 11
	mean_kernel = np.ones((mean_kernel_size,mean_kernel_size), np.float32)/(mean_kernel_size**2)
	blur_times = 2

	bgblur_kernel = np.ones((9,9), np.float32)/(9**2)

	grabbed = False
	mask2 = None

	T = time()
	timer = 0
	timer_steps = 10 # To calculate FPS

	while(True):

		c+=1

		img = cap.getCapture()
		loc = cap.getFace(img)

		if loc is not None:

			# Grab the head location
			(x, y, w, h) = loc

			# Draw the keypoints
			# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
			

			if(c%10==0):
				X=int(x/scale)
				Y=int(y/scale)
				W=int(w/scale)
				H=int(h/scale)
				smallimg = cv2.resize(img, (sW,sH), interpolation=cv2.INTER_LINEAR)
				rect = (max(1,X-int(W)),max(1,Y-int(H)),min(int(3*W),sW),smallimg.shape[0]-(Y-int(H)))

				initer = cv2.GC_INIT_WITH_RECT
				cap.mm, cap.bgdModel, cap.fgdModel = cv2.grabCut(smallimg,cap.mm,rect,cap.bgdModel,cap.fgdModel,5,initer)
				mask2 = np.where((cap.mm==2)|(cap.mm==0),0,1).astype('uint8')

				grabbed = True
				mask2 = cv2.resize(mask2, (wid,hei), interpolation=cv2.INTER_LINEAR)
				for _ in range(blur_times):
					mask2 = cv2.filter2D(mask2,-1,mean_kernel)


			if(grabbed):

				if(mode==None):

					pass

				elif (mode=='blur'):

					bgimg = cv2.resize(img, (int(wid/4), int(hei/4)))
					bgimg = cv2.filter2D(bgimg,-1,bgblur_kernel)
					bgimg = cv2.resize(bgimg, (wid,hei))

					img = img*mask2[:,:,np.newaxis] + bgimg*(1-mask2[:,:,np.newaxis])

				elif (mode=='remove'):

					img = img*mask2[:,:,np.newaxis] + bgimg*(1-mask2[:,:,np.newaxis])					

				else:

					raise NotImplementedError('Supported modes = ["blur", "remove", None]')

				pass

			# Show the image
			cv2.imshow('camera', img)

		else:
			cv2.imshow('camera', img)
			pass

		if cv2.waitKey(1) == 27:
			break  # esc to quit




		t = time()
		timer += t-T

		if(c%timer_steps==0):
			print("TIME: {} | FPS: {}".format(timer/timer_steps, timer_steps/timer))
			timer = 0

		T = t
