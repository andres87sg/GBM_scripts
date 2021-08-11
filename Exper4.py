#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:52:52 2021

@author: usuario
"""

import os
import random

import math
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import keras.backend as T

#%%
experiment_name = 'LungInf_SF2_Filt64_25052021'
scale_factor = 8
n_filters = 32
image_size = 512
batch_size = 32

img_width =  np.uint16(image_size/scale_factor)
img_height = np.uint16(image_size/scale_factor)
img_channels = 3
color_mode = "rgb"

input_dir = '/home/usuario/Documentos/LungInf/DataPartition/Train/CT/'
mask_dir = '/home/usuario/Documentos/LungInf/DataPartition/Train/Mask_M/'

val_dir = '/home/usuario/Documentos/LungInf/DataPartition/Val/CT/'
mask_val_dir = '/home/usuario/Documentos/LungInf/DataPartition/Val/Mask_M/'

# input_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/PC/'
# mask_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/Mask/'

# val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/PC/'
# mask_val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/Mask/'



target_size=(img_width, img_height)

image_datagen = ImageDataGenerator(
                                  rescale=1./255,
                                  rotation_range=25,                                  
                                  #fill_mode='nearest',
                                  #zoom_range=0.05,
                                 
                                  #width_shift_range=0.01,
                                  #height_shift_range=0.01,
                                  #horizontal_flip=True,
                                  #vertical_flip=True,
                                  #channel_shift_range=0.5,
                                  )


image_datagen_val = ImageDataGenerator(rescale=1./255)
mask_datagen_val = ImageDataGenerator(rescale=1./255)

image_generator = image_datagen.flow_from_directory(
    input_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size,  batch_size=batch_size,
    shuffle='True',seed=1)

mask_generator = image_datagen.flow_from_directory(
    mask_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    shuffle='True',seed=1)

image_generator_val = image_datagen_val.flow_from_directory(
    val_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    seed=1)

mask_generator_val = mask_datagen_val.flow_from_directory(
    mask_val_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    seed=1)

steps = image_generator.n//image_generator.batch_size
steps_val = image_generator_val.n//image_generator_val.batch_size


train_generator = zip(image_generator, mask_generator)
val_generator = zip(image_generator_val, mask_generator_val)

#%%

import random
n = random.randint(0,batch_size-1)


im1=image_generator_val[0]
mask1=mask_generator_val[0]

im1=im1[n,:,:,0]
mask=mask1[n,:,:,0]

plt.subplot(1,2,1)
plt.imshow(im1,cmap='gray')
plt.title('im_gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask,cmap='gray')
plt.title('mask')
plt.axis('off')

#%%

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet(input_img, n_filters, dropout, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


#%%

def soft_dice(y_pred, y_true):
    
    intersect = T.sum(y_pred * y_true, 0)
    denominator = T.sum(y_pred, 0) + T.sum(y_true, 0)
    dice_scores = T.constant(1)-T.constant(2) * intersect / (denominator + T.constant(1e-6))
    
    return dice_scores

def IoULoss(y_pred, y_true):
    
    smooth = T.constant(1e-6)
    intersection = T.sum(T.dot(y_pred,y_true))
    total = T.sum(y_true) + T.sum(y_pred)
    union = total - intersection
    IoU = (intersection+smooth) / (union+smooth)
    IoULossVal = T.constant(1)-IoU
    
    return IoULossVal

#%%

input_img = Input((img_width, img_height, img_channels ), name='img')
model = get_unet(input_img, n_filters=8, dropout=0.01, batchnorm=False)
#model.compile(optimizer='Adam', loss=soft_dices)
model.summary()

#%%
def step_decay(epoch):
	initial_lrate = 1e-3
	drop = 0.1
	epochs_drop = 50
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=20,mode='min', verbose=1)
checkpoint_path ='/home/usuario/Documentos/GBM/Experimentos/LungSegmModel.h5'

mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1 , save_best_only=True, mode='min')

model.compile(optimizer='adam',
              loss=soft_dice
              )

#%%

history = model.fit(train_generator,
                    steps_per_epoch=steps,
                    validation_data=val_generator,
                    validation_steps=steps_val,
                    epochs=200,
                    verbose=1,
                    callbacks=[es,mc,lr]
                    )


# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

#%%