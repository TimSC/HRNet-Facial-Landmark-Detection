import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config
from lib.datasets import get_dataset
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.losses import MeanSquaredError

import skimage.transform

class Object(object):
    pass

if __name__=="__main__":

	args = Object()
	args.cfg = 'experiments/300w/face_alignment_300w_hrnet_w18.yaml'
	update_config(config, args)
	dataset_type = get_dataset(config)
	dataset=dataset_type(config, is_train=True)

	#print (len(dataset))

	#for d in validation_data=(X_test, y_test):
	#	imgPlanes, target, meta = d
	#	print (imgPlanes.shape, target.shape)

	print ("Preparing model")
	model = ResNet50(
		include_top=False,
		#weights="imagenet",
		input_tensor=None,
		input_shape=(256, 256, 3),
		pooling='avg',
	)

	model.compile(optimizer='Adam', loss='mse')

	print ("Preparing training data")
	X_train = []
	y_train = []
	for i, samp in enumerate(dataset):
		imgPlanes, target, meta = samp
		if i%100==0:
			print (i, len(dataset), imgPlanes.shape, target.shape)
			#if i > 0:
			#	break

		img2 = preprocess_input(np.dstack((imgPlanes[0,:,:], imgPlanes[1,:,:], imgPlanes[2,:,:])))
		X_train.append(img2)

		#target2 = np.dstack([target[i,:,:] for i in range(target.shape[0])])
		target2 = np.zeros(2048, dtype=np.float32)
		target2[:45*45] = skimage.transform.resize(target[0,:,:], (45,45), anti_aliasing=True).flatten()
		y_train.append(target2)

	print ("Training")
	X_train = np.array(X_train)
	print (X_train.shape)
	y_train = np.array(y_train)
	print (y_train.shape)
	model.fit(X_train, y_train, batch_size=32, epochs=60, verbose=1) # validation_data=(X_test, y_test)
	
	model.save('kerasmodel')


