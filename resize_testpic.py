import cv2
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# img = cv2.imread('testimage/Selection_005.png')
# img_resized = cv2.resize(img,(28,28))
# 
# cv2.imshow('resized',img_resized)
# cv2.imshow('img',img)
# cv2.waitKey(0)

# iterate through all files in the directory to resize
# and convert image to grayscale



                
def resize_and_gray(path, width, height):
    subdirs = [x[0] for x in os.walk(path)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                print file
                img = cv2.imread(subdir + "/" + file)
                img_resized = cv2.resize(img, (width,height)) # width * height
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(subdir + "/" + file, img_gray)

resize_and_gray("testimage", 28,15)