import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt

img = cv.imread('SCIgen_sample_page_1.png',0)

#img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.resize(th3,(360,480))

plt.subplot(2,2,1),plt.imshow(th3,'gray')
plt.xticks([]),plt.yticks([])
plt.show()

