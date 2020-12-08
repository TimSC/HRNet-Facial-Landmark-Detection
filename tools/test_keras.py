
from tensorflow import keras

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

	pred = model.predict(X_test)

	print (pred.shape)

