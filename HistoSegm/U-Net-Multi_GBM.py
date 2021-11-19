# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:04:44 2021

@author: Andres
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


import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


#%%

def Jaccard_Metric(y_true, y_pred):
    delta = K.constant(1e-6)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) 
    Jaccard_Coef = (intersection + delta) / (union - intersection + delta)
    return Jaccard_Coef 

def Jaccard_Loss(y_true, y_pred):
    delta = K.constant(1e-6)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) 
    JaccardLossCoef = K.constant(1)-(intersection + delta) / (union - intersection + delta)
    return JaccardLossCoef 

def IoULoss(y_pred, y_true):
    
    # smooth = T.constant(1e-6)
    # intersection = T.sum(T.dot(y_pred,y_true))
    # total = T.sum(y_true) + T.sum(y_pred)
    # union = total - intersection
    # IoU = (intersection+smooth) / (union+smooth)
    # IoULossVal = T.constant(1)-IoU
    intersect = K.sum(y_pred * y_true, 0)
    total = K.sum(y_pred, 0) + K.sum(y_true, 0)
    union = total - intersect
    IoU = (intersect+ K.constant(1e-6)) / (union + K.constant(1e-6))
    IoULossVal = K.constant(1)-IoU
    
    #IoULossVal = T.constant(1)-T.constant(2) * intersect / (denominator + T.constant(1e-6))    
    
    
    return IoULossVal


    # total = T.sum(y_pred, 0) + T.sum(y_true, 0)
    # union = total - intersect
    # IoU = (intersect+ T.constant(1e-6)) / (union + T.constant(1e-6))

#%%
input_dir = '/home/usuario/Descargas/destino 10/Training/PC/'
mask_dir = '/home/usuario/Descargas/destino 10/Training/PC_SG2/'

val_dir = '/home/usuario/Descargas/destino 10/Validation/PC/'
mask_val_dir = '/home/usuario/Descargas/destino 10/Validation/PC_SG2/'

# input_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/MedSegData/CT/'
# mask_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/MedSegData/Mask2/'

# val_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/Validation/CT2/'
# mask_val_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/Validation/Mask2/'

# input_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/PC/'
# mask_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/Mask/'

# val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/PC/'
# mask_val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/Mask/'

scale_factor = 4
n_filters = 8
image_size = 2048
batch_size = 4

#%%

img_width =  np.uint16(image_size//scale_factor)
img_height = np.uint16(image_size//scale_factor)
img_channels = 3
color_mode = "rgb"


target_size=(img_width, img_height)

#%%

image_datagen = ImageDataGenerator(
                                  rescale=1./255,
                                  #validation_split=0.3,
                                  #rotation_range=2,                                  
                                  #fill_mode='nearest',
                                  #zoom_range=0.2,
                                 
                                  #width_shift_range=0.01,
                                  #height_shift_range=0.01,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  channel_shift_range=0.5,
                                  )

mask_datagen = ImageDataGenerator(
                                  rescale=1./255,
                                  #validation_split=0.3,
                                  #rotation_range=2,                                  
                                  #fill_mode='nearest',
                                  #zoom_range=0.2,
                                 
                                  #width_shift_range=0.01,
                                  #height_shift_range=0.01,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  channel_shift_range=0.5,
                                  )
  


#image_datagen_val = ImageDataGenerator(rescale=1./255,validation_split=0.2)
#mask_datagen_val = ImageDataGenerator(rescale=1./255,validation_split=0.2)
image_datagen_val = ImageDataGenerator(rescale=1./255)
mask_datagen_val = ImageDataGenerator(rescale=1./255)



image_generator = image_datagen.flow_from_directory(
    input_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size,  batch_size=batch_size,
    shuffle='False',seed=1)
    #,subset='training')

mask_generator = mask_datagen.flow_from_directory(
    mask_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    shuffle='False',seed=1)
    #subset='training')

image_generator_val = image_datagen_val.flow_from_directory(
    val_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    seed=1)
    #subset='validation')

mask_generator_val = mask_datagen_val.flow_from_directory(
    mask_val_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    seed=1)
    #subset='validation')

steps = image_generator.n//image_generator.batch_size
steps_val = image_generator_val.n//image_generator_val.batch_size


train_generator = zip(image_generator, mask_generator)
val_generator = zip(image_generator_val, mask_generator_val)

#%%

import matplotlib.pyplot as plt

import random
n = random.randint(0,batch_size-1)

im1=image_generator[0]
mask1=mask_generator[0]

im1=im1[n,:,:,:]
mask=mask1[n,:,:,:]

plt.subplot(1,2,1)
plt.imshow(im1,cmap='gray')
plt.title('im_gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask)
plt.title('mask')

plt.axis('off')


#%%

def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def build_unet(shape, num_classes):
    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    return Model(inputs, output)

model = build_unet((img_width, img_height, 3), 3)
model.summary()


def step_decay(epoch):
	initial_lrate = 1e-4
	drop = 0.1
	epochs_drop = 100
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=300,mode='min', verbose=1)
checkpoint_path ='/home/usuario/Documentos/GBM/Experimentos/InfSegmModel-Multi_128x128.h5'

mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1 , save_best_only=True, mode='min')


lr=0.001
#model.compile(loss="categorical_crossentropy", optimizer='Adam',metrics=['accuracy',Jaccard_Metric])
#model.compile(loss="categorical_crossentropy", optimizer='Adam',metrics=[Jaccard_Metric])

model.compile(loss=IoULoss, optimizer='Adam',metrics=[Jaccard_Metric])

history = model.fit(train_generator,
                    steps_per_epoch=steps,
                    validation_data=val_generator,
                    validation_steps=steps_val,
                    epochs=1,
                    verbose=1,
                    callbacks=[es,mc]
                    )


#%%

# model.load_weights('/home/usuario/Documentos/GBM/Experimentos/InfSegmModel-Multi.h5')

#%%

import cv2 as cv2
path = '/home/usuario/Descargas/destino 10/Training/PC/PC/'
test_path = '/home/usuario/Descargas/destino 10/Training/PC_SG2/PC_SG2/'
# path = '/home/usuario/Documentos/LungInf/NuevoLungInfDataset/CT2/'
# test_path = '/home/usuario/Documentos/LungInf/NuevoLungInfDataset/Mask3/'

imsize=512

# path = '/home/usuario/Documentos/LungInf/DataPartition/Val/CT/CT_png/'
# test_path = '/home/usuario/Documentos/LungInf/DataPartition/Val/Mask_M/Mask_png/'

listfiles = sorted(os.listdir(path))
mask_listfiles = sorted(os.listdir(test_path))


for i in range(19,20):

  # List of files
  mask_im_name = mask_listfiles[i]
  im_name = listfiles[i]
       
# Groundtruth image (array)
  mask_array=cv2.imread(test_path+mask_im_name)   # Mask image
  im_array = cv2.imread(path+im_name)               # Graylevel image
  
  kk=mask_array.copy()
  
  
  #im_gray = im_array.copy()
  im_gray = im_array
  #mask_array.shape()
  # Groundtruth mask Image resize
  mask_array=cv2.resize(mask_array,(imsize,imsize),interpolation = cv2.INTER_AREA)
  #print(np.unique(kk))
  ## Input image to model must be 128x128 therefore 512/4
  scale = 4
  
  
  # Image resize must resize (Model input 128 x 128)
  im_array=cv2.resize(im_array,(imsize,imsize),
                      interpolation = cv2.INTER_AREA)
  im_array=im_array/255
  
  # Adding one dimension to array
  img_array = np.expand_dims(im_array,axis=[0])
  # Generate image prediction
  pred_mask = model.predict(img_array)
  
  pred_mask2 = pred_mask.copy()
  

  #zzz=pred_mask.copy()
  
  zzz=pred_mask[0,:,:,0]
  zzz1=pred_mask[0,:,:,:]
  zzz2=pred_mask.copy()
  nn=np.argmax(zzz1,axis=-1)
  plt.imshow(nn)
  plt.title('eso')
  
  
  # Image mask as (NxMx1) array
  pred_mask = pred_mask[0,:,:,2]
  pred_mask = np.uint16(np.round(pred_mask>0.1))
  
  # Resize image to 512x512x1
  pred_mask = cv2.resize(pred_mask,(imsize,imsize), 
                      interpolation = cv2.INTER_NEAREST)
  
  true_mask = np.uint16(mask_array[:,:,0])//255
  
  plt.figure()
  plt.subplot(1,3,1)
  plt.imshow(im_array[:,:,:])
  plt.title('Sample')
  plt.axis('off')
  plt.subplot(1,3,2)
  plt.imshow(mask_array[:,:,0])
  plt.title('Ground truth')
  plt.axis('off')
  plt.subplot(1,3,3)
  plt.imshow(pred_mask)
  plt.title('Prediction')
  plt.axis('off')

#%%

import skimage.segmentation
from skimage.segmentation import mark_boundaries

plt.figure()
plt.imshow(mark_boundaries(img_array[0,:,:,0],pred_mask))

#%%



