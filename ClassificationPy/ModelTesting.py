# -*- coding: utf-8 -*-
"""
@author: Andres Sandino

https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class

"""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Input,layers, models
from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

    

#%%
image_size = 512
scale_factor = 2

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
    conv5 = Dropout(0.5)(conv5)
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

####### Model SETUP ###########
model = Unet(image_size//scale_factor, image_size//scale_factor, nclasses=2, filters=8)

#model.summary()


#%%
dir='C:/Users/Andres/Desktop/SegmModel2.h5'
model=keras.models.load_model(dir)
model.summary()

#Loading model weights
#model.load_weights(dir)

#%%

def imoverlay(img,predimg,coloredge):
    
    
    #predictedrgb=np.zeros((512,512,3))
    
    # Upsample image to 512x512
    predmask = cv2.resize(predimg,(512,512), interpolation = cv2.INTER_AREA)
    predmask = predmask*255
    
    # Upsample image to 512x512    
    img = cv2.resize(img,(512,512), interpolation = cv2.INTER_AREA)
    
    overlayimg = img.copy()
    overlayimg[predmask == 255] = coloredge
    
    return overlayimg

#%% Visualizacion de resultados (No es necesario correr esta sección)

#C:\Users\Andres\Desktop\imexhs\Lung\dicomimage\Torax\dcm2png\nuevos_casos_train
#path = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/dcm2png/test_dcm/'
#def displayresults():

path = 'C:/Users/Andres/Desktop/segm/test11/HB/'
listfiles = os.listdir(path)



#for i in range(len(listfiles)):
for i in range(0,5):
    
    # List of files
    im_name = listfiles[i]
    print(im_name)
    
    # Graylevel image (array)
    im_array=cv2.imread(path+im_name)
    print(im_array.shape)
    
    scale = 2
    
    # Image resize
    im_array=cv2.resize(im_array,(512//scale,512//scale), 
                        interpolation = cv2.INTER_AREA)
    
    # Image gray level normalization
    im_array=im_array/255
    
    # Adding one dimension to array
    img_array = np.expand_dims(im_array,axis=[0])
    
    # Generate image prediction
    pred_mask = model.predict(img_array)
    
    # Image mask as (NxMx1) array
    pred_mask = pred_mask[0,:,:,0]
    pred_mask=np.uint16(np.round(pred_mask>0.35))
    
    # Image overlay (mask - gray level) (Visualization)
    pred=imoverlay(im_array,pred_mask,[255,0,0])
    
#    plt.imshow(pred_mask,cmap='gray')
    plt.imshow(pred,cmap='gray')
    #plt.title('Predicted mask'+im_name)
    plt.title(im_name)
    plt.axis('off')     
    plt.show()
    plt.close()

#displayresults()
    
#%% Compute Metrics

print('Computing Metrics...')

path = 'C:/Users/Andres/Desktop/segm/test11/HB/'
test_path = 'C:/Users/Andres/Desktop/segm/test11/HB_SG/'

listfiles = os.listdir(path)
mask_listfiles = os.listdir(test_path)

dicescore = []
accuracy = []
sensitivity = []
specificity = []
f1score = []
    

#for i in range(len(listfiles)):
for i in range(0,100):
  
    
    # List of files
    mask_im_name = mask_listfiles[i]
    im_name = listfiles[i]
       
    # Groundtruth image (array)
    mask_array=cv2.imread(test_path+mask_im_name)   # Mask image
    im_array=cv2.imread(path+im_name)               # Graylevel image
    
    # Groundtruth mask Image resize
    mask_array=cv2.resize(mask_array,(512,512),interpolation = cv2.INTER_AREA)
    
    ## Input image to model must be 128x128 therefore 512/4
    scale = 2
    
    # Image resize must resize (Model input 128 x 128)
    im_array=cv2.resize(im_array,(512//scale,512//scale),interpolation = cv2.INTER_AREA)
    im_array=im_array/255
    
    # Adding one dimension to array
    img_array = np.expand_dims(im_array,axis=[0])
    
    # Generate image prediction
    pred_mask = model.predict(img_array)
    
    # Image mask as (NxMx1) array
    pred_mask = pred_mask[0,:,:,0]
    pred_mask = np.uint16(np.round(pred_mask>0.5))
    
    # Resize image to 512x512x1
    pred_mask = cv2.resize(pred_mask,(512,512), 
                        interpolation = cv2.INTER_AREA)
    
    true_mask = np.uint16(mask_array[:,:,0])//255
    
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
    f1 = 2*tp/(2*tp+fp+fn)
    
    accuracy.append(acc)
    sensitivity.append(sens)
    specificity.append(spec)
    f1score.append(f1)
    

# Metrics

dicescore = np.array(dicescore)
meandice = np.mean(dicescore)
stddice = np.std(dicescore)

accuracy = np.array(accuracy)
meanacc = np.mean(accuracy)
stdacc = np.std(accuracy)

f1sco = np.array(f1score)
meanf1 = np.mean(f1sco)
stdf1 = np.std(f1sco)

print('------------------------')    
print('Mean Dice: '+str(meandice))
print('Std Dice: '+str(stddice))
print('------------------------')
print('Mean Acc: '+str(meanacc))
print('Std Acc: '+str(stdacc))
print('------------------------')
print('Mean F1: '+str(meanf1))
print('Std F1: '+str(stdf1))   
print('------------------------')    
    

    



  
