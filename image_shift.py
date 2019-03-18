#this file moves the images from one folder to other renaming them as img-0-(1,2,3..i).png

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import glob

#source folder of images
image=[cv.imread(file,0) for file in glob.glob("path/to/img-0*.png")]
print(len(image))
print(image.shape)

for i in range(2):
    img=image[i]
    print(type(img))
    #destination folder of images 
    path='my_dataset/raw/'
    cv.imwrite(os.path.join(path , 'img-0-'+str(i)+'.png'), img)
    cv.waitKey(0) 
    cv.imshow('gray',img)     
    cv.destroyAllWindows()
    cv.waitKey(1)
