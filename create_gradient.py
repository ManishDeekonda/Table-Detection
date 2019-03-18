import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
import os
import glob

def gradient(image):
    #img = cv.imread(image,0)   #read the image in 2d
    img=image

    blur = cv.GaussianBlur(img,(3,3),0)         #filter the image blur,threshold,denoise
    ret,img = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    cv.fastNlMeansDenoising(img,None,10,7,21)
    """
    kernel = np.ones((3,3),np.uint8)              #erosion then dilation for increasing clear white spaces   
    erode = cv.erode(img,kernel,iterations = 1)
    #kernel = np.ones((3,3),np.uint8)
    img = cv.dilate(erode,kernel,iterations = 1)
    """
    kernel = np.ones((1,1),np.uint8)                                        
    opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel, iterations = 2) #apply distance transform using opencv
    img = cv.distanceTransform(img,cv.DIST_MASK_3,5)            
    
    
    return img

image=[]
#image=cv.imread("10.1.1.1.2023_31.bmp")

#take care of (j=0/1) img-0 is for english images and img-1 for chinese
j=0 #change j=1 and run the file again for rest of images
for i in range(508):
    #source images path..
    image=[cv.imread(file,0) for file in glob.glob("my_dataset/raw/img-"+str(j)+"-"+str(i)+".png")]
    img=gradient(image[0])
    #destination path
    path = 'my_dataset/trimaps/'
    #img.save(os.path.join(path , 'img0.png'))
    cv.imwrite(os.path.join(path , 'img-0-'+str(i)+'.png'), img)
    cv.waitKey(0) 
    cv.imshow('gray',img)                  #display the image
    cv.destroyAllWindows()
    cv.waitKey(1)
