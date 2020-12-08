import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core.evaluation import decode_preds

import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

class Object(object):
    pass

if __name__=="__main__":
	model = keras.models.load_model('kerasmodel')

	args = Object()
	args.cfg = 'experiments/300w/face_alignment_300w_hrnet_w18.yaml'
	update_config(config, args)
	dataset_type = get_dataset(config)
	dataset=dataset_type(config, is_train=False)

	X_test = []

	for i, samp in enumerate(dataset):
		imgPlanes, target, meta = samp
		if i%10==0:
			print (i, len(dataset), imgPlanes.shape, target.shape)
			if i > 0:
				break

		img2 = preprocess_input(np.dstack((imgPlanes[0,:,:], imgPlanes[1,:,:], imgPlanes[2,:,:])))
		X_test.append(img2)

	X_test = np.array(X_test)
	print (X_test.shape)

	pred = model.predict(X_test)

	print (pred.shape)

	for i in range(pred.shape[0]):
		img = pred[i, :]
		img = img[:45*45].reshape((45, 45))
		plt.imshow(img)
		plt.show()

	#preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])


