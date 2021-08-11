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
#experiment_name = 'LungInf_SF2_Filt64_25052021'
scale_factor = 4
n_filters = 32
image_size = 512
batch_size = 32

img_width =  np.uint16(image_size/scale_factor)
img_height = np.uint16(image_size/scale_factor)
img_channels = 1
color_mode = "rgb"

# input_dir = '/home/usuario/Documentos/LungInf/DataPartition/Train/CT/'
# mask_dir = '/home/usuario/Documentos/LungInf/DataPartition/Train/Mask_M/'

# val_dir = '/home/usuario/Documentos/LungInf/DataPartition/Val/CT/'
# mask_val_dir = '/home/usuario/Documentos/LungInf/DataPartition/Val/Mask_M/'

#input_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/Training/CT2/'
#mask_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/Training/Mask2/'

input_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/MedSegData/CT/'
mask_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/MedSegData/Mask/'


val_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/Validation/CT2/'
mask_val_dir = '/home/usuario/Documentos/LungInf/LungInfDataset/Validation/Mask2/'


# input_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/PC/'
# mask_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/Mask/'

# val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/PC/'
# mask_val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/Mask/'



target_size=(img_width, img_height)

image_datagen = ImageDataGenerator(
                                  rescale=1./255,
                                  #rotation_range=5,                                  
                                  #fill_mode='nearest',
                                  #zoom_range=0.2,
                                 
                                  #width_shift_range=0.01,
                                  #height_shift_range=0.01,
                                  #horizontal_flip=True,
                                  #vertical_flip=True,
                                  #channel_shift_range=0.5,
                                  )


image_datagen_val = ImageDataGenerator(rescale=1./255,validation_split=0.3)
mask_datagen_val = ImageDataGenerator(rescale=1./255,validation_split=0.3)
# image_datagen_val = ImageDataGenerator(rescale=1./255)
# mask_datagen_val = ImageDataGenerator(rescale=1./255)



image_generator = image_datagen.flow_from_directory(
    input_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size,  batch_size=batch_size,
    shuffle='False',seed=1)

mask_generator = image_datagen.flow_from_directory(
    mask_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    shuffle='False',seed=1)

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


im1=image_generator[0]
mask1=mask_generator[0]

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

nclasses=4

model = Unet(512//scale_factor,512//scale_factor, nclasses, filters=32)

model.summary()
a=0

#%%

def soft_dice(y_pred, y_true):
    
    intersect = T.sum(y_pred * y_true, 0)
    denominator = T.sum(y_pred, 0) + T.sum(y_true, 0)
    dice_scores = T.constant(1)-T.constant(2) * intersect / (denominator + T.constant(1e-6))
    
    return dice_scores

def IoULoss(y_pred, y_true):
    
    # smooth = T.constant(1e-6)
    # intersection = T.sum(T.dot(y_pred,y_true))
    # total = T.sum(y_true) + T.sum(y_pred)
    # union = total - intersection
    # IoU = (intersection+smooth) / (union+smooth)
    # IoULossVal = T.constant(1)-IoU
    intersect = T.sum(y_pred * y_true, 0)
    total = T.sum(y_pred, 0) + T.sum(y_true, 0)
    union = total - intersect
    IoU = (intersect+ T.constant(1e-6)) / (union + T.constant(1e-6))
    IoULossVal = T.constant(1)-IoU
    
    #IoULossVal = T.constant(1)-T.constant(2) * intersect / (denominator + T.constant(1e-6))    
    
    
    return IoULossVal


#%%
def step_decay(epoch):
	initial_lrate = 1e-4
	drop = 0.1
	epochs_drop = 100
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=30,mode='min', verbose=1)
checkpoint_path ='/home/usuario/Documentos/GBM/Experimentos/InfSegmModel.h5'

mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1 , save_best_only=True, mode='min')

model.compile(optimizer='adam',
              #loss=IoULoss,
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics='accuracy'
              )

#%%

history = model.fit(train_generator,
                    steps_per_epoch=steps,
                    validation_data=train_generator,
                    validation_steps=steps_val,
                    epochs=300,
                    verbose=1,
                    callbacks=[es,mc,lr]
                    )


# https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

#%%

import numpy as np

# plt.figure(1)
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.yticks(np.arange(0, 1, step=0.05))
# plt.grid(color='k', linestyle='--', linewidth=0.4)
# plt.legend(loc='lower right')
# #plt.savefig(dir + 'accuracy_CNN_ ' + Exp +  '.png')

plt.figure(2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 1])
#plt.yticks(np.arange(0, 1, step=0.05))
plt.grid(color='k', linestyle='--', linewidth=0.4)
plt.legend(loc='lower right')
#plt.savefig(dir + 'loss_CNN_' + Exp + '.png')



#%%
import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
#from tqdm import tqdm # Progress bar

import tensorflow as tf
import tensorflow.keras as keras

#model = keras.models.load_model('/home/usuario/Documentos/GBM/Experimentos/PCSegmModel.h5')

#model.load_weights('/home/usuario/Documentos/GBM/Experimentos/LungSegmModel.h5')
model.load_weights('/home/usuario/Documentos/GBM/Experimentos/InfSegmModel.h5')
#model.save('/home/usuario/Documentos/GBM/Experimentos/InfSegmModelCompleto.h5')

    
    
    
    #%%

import cv2 as cv2
path = '/home/usuario/Documentos/LungInf/LungInfDataset/Testing/CT2/CT/'
test_path = '/home/usuario/Documentos/LungInf/LungInfDataset/Testing/Mask/Mask/'
# path = '/home/usuario/Documentos/LungInf/NuevoLungInfDataset/CT2/'
# test_path = '/home/usuario/Documentos/LungInf/NuevoLungInfDataset/Mask3/'



# path = '/home/usuario/Documentos/LungInf/DataPartition/Val/CT/CT_png/'
# test_path = '/home/usuario/Documentos/LungInf/DataPartition/Val/Mask_M/Mask_png/'

listfiles = sorted(os.listdir(path))
mask_listfiles = sorted(os.listdir(test_path))

dicescore = []
accuracy = []
sensitivity = []
specificity = []
f1score = []

IoUmetric = []

## Input image to model must be 128x128 therefore 512/4
#scale = 8
imsize = 128

for i in range(130,131):
#for i in range(31,39):
#for i in tqdm(range(100)):
#for i in range(2,3):

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
  print(np.unique(kk))
  ## Input image to model must be 128x128 therefore 512/4
  scale = 8
  
  # Image resize must resize (Model input 128 x 128)
  im_array=cv2.resize(im_array,(imsize,imsize),
                      interpolation = cv2.INTER_NEAREST)
  im_array=im_array/255
  
  # Adding one dimension to array
  img_array = np.expand_dims(im_array,axis=[0])
  # Generate image prediction
  pred_mask = model.predict(img_array)
  
  #zzz=pred_mask.copy()
  
  zzz=pred_mask[0,:,:,0]
  
  # Image mask as (NxMx1) array
  pred_mask = pred_mask[0,:,:,0]
  pred_mask = np.uint16(np.round(pred_mask>0.99))
  
  # Resize image to 512x512x1
  pred_mask = cv2.resize(pred_mask,(imsize,imsize), 
                      interpolation = cv2.INTER_NEAREST)
  
  
  
  true_mask = np.uint16(mask_array[:,:,0])//255
  
  plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(mask_array[:,:,0])
  plt.title('Ground truth')
  plt.subplot(1,2,2)
  plt.imshow(pred_mask)
  plt.title('Prediction')
  
  intersectmask = true_mask & pred_mask
  
  #sumintersectmask = np.sum(intersectmask)
  
  sumpredtrue = np.sum(true_mask)+np.sum(pred_mask)
  
  if sumpredtrue != 0:
        
      dice = 2*np.sum(intersectmask)/(np.sum(true_mask)+np.sum(pred_mask)+0.001)
  
      dicescore.append(dice)
  
  true_mask_flat = true_mask.flatten()
  pred_mask_flat = pred_mask.flatten()
  
  p = np.sum(true_mask_flat)
  n = np.sum(np.logical_not(true_mask_flat))
  tp = np.sum(true_mask_flat & pred_mask_flat)
  fp = np.sum(np.logical_not(true_mask_flat) & pred_mask_flat)
  tn = np.sum(np.logical_not(true_mask_flat) & np.logical_not(pred_mask_flat))
  fn = np.sum(true_mask_flat & np.logical_not(pred_mask_flat))
  
  acc = (tp+tn)/(p+n)
  sens = tp/(tp+fn+0.01) # Cuidado BUG!
  spec = tn/(tn+fp)

  IoU = (tp+0.00001)/(tp+fp+fn+0.00001)

  #f1 = 2*tp/(2*tp+fp+fn)
  
  IoUmetric.append(IoU)
  accuracy.append(acc)
  sensitivity.append(sens)
  specificity.append(spec)
    #f1score.append(f1)

# Metrics

dicescore = np.array(dicescore)
meandice = np.mean(dicescore)
stddice = np.std(dicescore)

accuracy = np.array(accuracy)
meanacc = np.mean(accuracy)
stdacc = np.std(accuracy)

IoU = np.array(IoUmetric)
meanIoU = np.mean(IoUmetric)
stdIoU = np.std(IoUmetric)

# f1sco = np.array(f1score)
# meanf1 = np.mean(f1sco)
# stdf1 = np.std(f1sco)

print('------------------------')    
print('Mean Dice: '+str(meandice))
print('Std Dice: '+str(stddice))
print('------------------------')
print('Mean Acc: '+str(meanacc))
print('Std Acc: '+str(stdacc))
print('------------------------')
print('------------------------')
print('Mean IoU: '+str(meanIoU))
print('Std IoU: '+str(stdIoU))
print('------------------------')


import skimage.segmentation
from skimage.segmentation import mark_boundaries

plt.figure()
plt.imshow(mark_boundaries(img_array[0,:,:,0],pred_mask))
