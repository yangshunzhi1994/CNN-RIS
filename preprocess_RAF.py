# create data and label for CK+
#  0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=contempt
# contain 135,177,75,207,84,249,54 images

import csv
import os
import numpy as np
import h5py
import skimage.io
import cv2 as cv

train_path = '/home/ysz/Mask_RCNN/data/pytoch/Expression13/data/RAF/train'
valid_path = '/home/ysz/Mask_RCNN/data/pytoch/Expression13/data/RAF/valid'

train_anger_path = os.path.join(train_path, '5')
train_disgust_path = os.path.join(train_path, '2')
train_fear_path = os.path.join(train_path, '1')
train_happy_path = os.path.join(train_path, '3')
train_sadness_path = os.path.join(train_path, '4')
train_surprise_path = os.path.join(train_path, '0')
train_contempt_path = os.path.join(train_path, '6')
# # Creat the list to store the data and label information
train_data_x = []
train_data_y = []

valid_anger_path = os.path.join(valid_path, '5')
valid_disgust_path = os.path.join(valid_path, '2')
valid_fear_path = os.path.join(valid_path, '1')
valid_happy_path = os.path.join(valid_path, '3')
valid_sadness_path = os.path.join(valid_path, '4')
valid_surprise_path = os.path.join(valid_path, '0')
valid_contempt_path = os.path.join(valid_path, '6')
# # Creat the list to store the data and label information
valid_data_x = []
valid_data_y = []

datapath = os.path.join('/home/ysz/Mask_RCNN/data/pytoch/Expression13/data','RAF_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(train_anger_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_anger_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(0)
print (1)
files = os.listdir(train_disgust_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_disgust_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(1)
print (1)
files = os.listdir(train_fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_fear_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(2)
print (1)
files = os.listdir(train_happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_happy_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(3)
print (1)
files = os.listdir(train_sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_sadness_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(4)
print (1)
files = os.listdir(train_surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_surprise_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(5)
print (1)
files = os.listdir(train_contempt_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_contempt_path,filename))
    I = cv.resize(I, (48, 48))
    train_data_x.append(I.tolist())
    train_data_y.append(6)

print(np.shape(train_data_x))
print(np.shape(train_data_y))


# order the file, so the valid set will not contain the test set (don't random)
files = os.listdir(valid_anger_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_anger_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(0)

files = os.listdir(valid_disgust_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_disgust_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(1)

files = os.listdir(valid_fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_fear_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(2)

files = os.listdir(valid_happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_happy_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(3)

files = os.listdir(valid_sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_sadness_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(4)

files = os.listdir(valid_surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_surprise_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(5)

files = os.listdir(valid_contempt_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_contempt_path,filename))
    I = cv.resize(I, (48, 48))
    valid_data_x.append(I.tolist())
    valid_data_y.append(6)

print(np.shape(valid_data_x))
print(np.shape(valid_data_y))



datafile = h5py.File(datapath, 'w')
datafile.create_dataset("train_data_pixel", dtype = 'uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype = 'int64', data=train_data_y)
datafile.create_dataset("valid_data_pixel", dtype = 'uint8', data=valid_data_x)
datafile.create_dataset("valid_data_label", dtype = 'int64', data=valid_data_y)
datafile.close()

print("Save data finish!!!")
