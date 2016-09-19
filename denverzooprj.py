
'''Trains a simple convnet on the sorted Denver Zoo dataset.
Gets to 75% \\test accuracy after only 5 epochs
(there is still a lot of margin for parameter tuning).
Relatively slow on CPU only machine, around 10min to complete 3 epochs.

Author: Qi Liu
'''

from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# image loading
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import re
import timeit

# training parameters
batch_size = 20 #128
nb_classes = 3 #10
nb_epoch = 1 #iterations to run, increase to improve accuracy, take much longer time

# input image dimensions
img_rows, img_cols = 200, 200
color_dim = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

dpath = '/home/mirabot/Documents/deeplearning/zoophoto/zoosorted/mixed_resize/'
labelpath = '/home/mirabot/Documents/deeplearning/zoophoto/zoosorted/label.csv'

imagePath = sorted(glob.glob(dpath+'*.JPG'),key=numericalSort)
im_array = np.array( [np.array(Image.open(imagePath[i]).convert('RGB'), 'f') for i in range(len(imagePath))])
print (im_array.shape)

# read in groundtruth label
label = np.genfromtxt(labelpath, dtype=float, delimiter=',', names=True)
np.random.seed(0)# for consistency

image_n = label.size
ratio = 0.85 # percentage of training data
train_id = np.random.choice(image_n, size=(int(image_n*ratio), 1), replace=False)
image_id = np.linspace(0, image_n-1, num=image_n, endpoint=True, dtype='int')
test_id = np.setdiff1d(image_id,train_id)

# training and testing data set
X_train = im_array[train_id,:,:,:]
Y_train = label[train_id]
X_test = im_array[test_id,:,:,:]
Y_test = label[test_id]

plt.imshow(im_array[883,:,:,:],cmap="gray")

"""TO-DO: Random sample training and testing sets with labels, make radom seed consistent
Then train 1 image to test, see how much time, then train 80% \\of the data sets
Then save model weights to h5
Make this procedure automatic"""

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], color_dim, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], color_dim, img_rows, img_cols)
    input_shape = (color_dim, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, color_dim)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, color_dim)
    input_shape = (img_rows, img_cols, color_dim)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)
# timing profile
start_time = timeit.default_timer()
# do things below
# build and train model
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# end of things, profile
# code you want to evaluate
elapsed = timeit.default_timer() - start_time
print ("Elapsed time: %s s." % (elapsed))
