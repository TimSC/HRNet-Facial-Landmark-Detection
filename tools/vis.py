import torch
import os
import csv 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio

if __name__=="__main__":

    predictions = torch.load(os.path.join('output/300W/face_alignment_300w_hrnet_w18', 'predictions.pth'))

    print(predictions.shape)

    dataList = csv.DictReader(open("data/300w/face_landmarks_300w_valid.csv", "rt"))
    for pred, data in zip(predictions, dataList):
        print (data['image_name'])

        img = imageio.imread(os.path.join("data/300w/images", data['image_name']), as_gray=False, pilmode="RGB")

        for pt in pred:
            pt = [int(np.round(np.array(c))) for c in pt]
            img[pt[1], pt[0], 0] = 0
            img[pt[1], pt[0], 1] = 255
            img[pt[1], pt[0], 2] = 0

        plt.imshow(img)
        plt.show()

        
