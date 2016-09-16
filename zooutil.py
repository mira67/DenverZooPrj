"""Preprocess Images and Load
Author: Qi Liu Modified"""

import numpy as np
import json
import os

from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing import image as image_utils

CLASS_INDEX = None
CLASS_INDEX_PATH = None # TBD


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds):
    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 1000
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    indices = np.argmax(preds, axis=-1)
    results = []
    for i in indices:
        results.append(CLASS_INDEX[str(i)])
    return results

def reformat_input(imfile,m,n):
    # load the input image using the Keras helper utility while ensuring
    # that the image is resized to m x n pxiels, the required input
    # dimensions for the network -- then convert the PIL image to a
    # NumPy array
    image = image_utils.load_img(imfile, target_size=(m, n))
    image = image_utils.img_to_array(image)

    # our image is now represented by a NumPy array of shape (N, m, n)
    # N = 3 when RGB image, N =1 when gray image
    # but we need to expand the dimensions to be (1, N, m, n) so we can
    # pass it through the network -- we'll also preprocess the image by
    # subtracting the mean RGB pixel intensity from the ImageNet dataset
    # redesign this subtraction for Remote Sensing Data
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
