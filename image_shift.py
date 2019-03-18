import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
import os
import glob



#print(len(image))
image=[cv.imread(file,0) for file in glob.glob("my_dataset/test_images/*.jpg")]
print(len(image))


for i in range(8):
    img=image[i]
    print(type(img))
    path='my_dataset/test_images/'
    cv.imwrite(os.path.join(path , 'img-0-'+str(i)+'.jpg'), img)
    cv.waitKey(0) 
    cv.imshow('gray',img)     
    cv.destroyAllWindows()
    cv.waitKey(1)
