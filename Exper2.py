#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:52:52 2021

@author: usuario
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#%%
# Set some parameters
im_width = 128
im_height = 128
border = 5

#%%
input_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/PC/PC/'
mask_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/Mask/Mask/'

#%%
ids = next(os.walk(input_dir))[2] # list of names all images in the given path
print("No. of images = ", len(ids))

#%%
X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)

#%%
import cv2 as cv

for i in range(len(ids)):
    # Load images
    img = cv.imread(input_dir+ids[i])
    x_img = resized = cv.resize(img, (128,128), interpolation = cv.INTER_AREA)
    # img = load_img(input_dir+ids[i])
    x_img = img_to_array(x_img)
   # x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    # # Load masks
    mask = cv.imread(mask_dir+'SG_'+ids[i])
    x_mask = resized = cv.resize(mask, (128,128), interpolation = cv.INTER_AREA)
    x_mask = img_to_array(np.uint16(x_mask/255))
    # mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
    # # Save images
    X[i] = x_img[:,:,:]/255.0
    y[i] = x_mask
    
#%%
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, shuffle=False)




#%%


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = tf.concat([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses, filters):
# down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5,name='BOTTLENECK')(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

nclasses=2
image_size=128
scale_factor=1
filters=32
model = Unet(image_size//scale_factor, image_size//scale_factor, nclasses, filters)

model.summary()


#%%

def soft_dice(y_pred, y_true):
    import keras.backend as T
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    #a=0
    intersect = T.sum(y_pred * y_true, 0)
    denominator = T.sum(y_pred, 0) + T.sum(y_true, 0)
    dice_scores = T.constant(1)-T.constant(2) * intersect / (denominator + T.constant(1e-6))
    #print("aaa")
    return dice_scores

#%%


#input_img = Input((im_height, im_width, 1), name='img')
#model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer='Adam', loss=soft_dice,)
#model.compile(optimizer='Adam', loss=soft_dice)
model.summary()


#%%

results = model.fit(X_train, y_train, batch_size=4, epochs=50,
                    validation_data=(X_valid, y_valid))