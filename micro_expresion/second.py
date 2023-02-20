import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys
import tensorflow as tf
from keras.models import load_model
import glob
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES']= '1'

K.image_data_format() == "channels_last"

image_rows, image_columns, image_depth = 64, 64, 18

training_list = []

negativepath = 'train_set/negative/'
positivepath = 'train_set/positive/'
surprisepath = 'train_set/surprise/'
otherpath = 'train_set/others/'

def cv_imread(filepath):
    cv_img = cv2.imdecode(numpy.fromfile(filepath,dtype = numpy.uint8),-1)
    return cv_img

directorylisting = os.listdir(negativepath)
count = 0
for video in directorylisting:
    count = count + 1
    videopath = negativepath + video
    #print(videopath)
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv_imread(imagepath)
           #print(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           #print(image.shape)
           
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
           
           frames.append(grayimage)
           
    frames = numpy.asarray(frames)
    
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)


directorylisting = os.listdir(positivepath)
count1 = 0
for video in directorylisting:
    count1 = count1 + 1
    videopath = positivepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv_imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
           
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

directorylisting = os.listdir(surprisepath)
count2 = 0
for video in directorylisting:
    videopath = surprisepath + video
    count2 = count2 + 1
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv_imread(imagepath)
           #print(frame)
           #print(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
           
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)


directorylisting = os.listdir(otherpath)
count3 = 0
for video in directorylisting:
    count3 = count3 + 1
    videopath = otherpath + video
    #print(videopath)
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv_imread(imagepath)
           #print(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           #print(image.shape)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    #videoarray2 = numpy.rollaxis(numpy.rollaxis(frames2, 2, 0), 2, 0)
    training_list.append(videoarray)
    #training_list.append(videoarray2)
    #print(training_list)

training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)

traininglabels = numpy.zeros((trainingsamples, ), dtype = int)

traininglabels[0:69] = 0
traininglabels[69:99] = 1
traininglabels[99:125] = 2
traininglabels[125:] = 3

traininglabels = np_utils.to_categorical(traininglabels, 4)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))


for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

# Spliting the dataset into training and test sets
train_images, test_images, train_labels, test_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)
train_images = train_images.reshape(train_images.shape[0], train_images.shape[4], train_images.shape[2], train_images.shape[3], train_images.shape[1])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[4], test_images.shape[2], test_images.shape[3], test_images.shape[1])
# MicroExpSTCNN Model
model = Sequential()
initializer = tf.random_normal_initializer(0., 0.02)
model.add(Conv3D(32, (3, 3, 15), kernel_initializer=initializer, input_shape=(image_depth, image_rows, image_columns,1), data_format='channels_last'))
model.add(tf.keras.layers.Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(tf.keras.layers.Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation = 'softmax'))
#model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(0.01), metrics = ['accuracy'])
model.summary()

filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Training the model
#hist = model.fit(train_images, train_labels, test_data = (test_images, test_labels), callbacks=callbacks_list, batch_size = 16, epochs = 10, shuffle=True)
hist = model.fit(train_images, train_labels, validation_data = (test_images, test_labels), batch_size = 16, epochs = 10)
# Finding Confusion Matrix using pretrained weights

#%%
'¦s¼Ò«¬&Åª¼Ò«¬'
model.save("detect_expression_model.h5")
#model = load_model('CNN_model')


