import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy import misc 

def create_batches(batch_size):
  images = []
  for img in list_of_images:
    images.append(misc.imread(img))
  images = np.asarray(images)


  while (True):
    for i in range(0,total,batch_size):
      yield(images[i:i+batch_size],labels[i:i+batch_size])
imgs = tf.placeholder(tf.float32,shape=[None,height,width,colors])
lbls = tf.placeholder(tf.int32, shape=[None,label_dimension])

with tf.Session() as sess:
#define rest of graph here
# convolutions or linear layers and cost function etc.


  batch_generator = create_batches(batch_size)
  for i in range(number_of_epochs):
    images, labels = batch_generator.next()
    loss_value = sess.run([loss], feed_dict={imgs:images, lbls:labels})
