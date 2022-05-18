# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:30:36 2020
Modified: 1 May 2022

@author: Andres
"""
#%% Import Libraries 

import math
import cv2 as cv
import numpy as np

import tensorflow.keras as keras
import tensorflow as tf

from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm


from matplotlib import pyplot as plt
import PIL 
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from ClassifierUtils import scaled_wsi, wsiregion, grtrpixelmask
from ClassifierUtils import pixtomask

import slideio
import cv2 as cv
import matplotlib.pyplot as plt

#%% 

wsi_name = 'TCGA-08-051'
wsi_path = '/home/usuario/Documentos/GBM/TCGA/'
mask_filename = wsi_name + '_mask.png' # Mask provided by HistoQC
wsi_filename = wsi_name + '.svs'

slide = slideio.open_slide(wsi_path + wsi_filename,"SVS")

wsi = slide.get_scene(0)
magnification = wsi.magnification
WSIheight,WSIwidth =wsi.size

# ROI Mask
mask = cv.imread(wsi_path + mask_filename)

th=0.5              # Tissue percentage
patchsize = 224     # Patch Size
scale = 2           # Downsampling factor

# Scaled image
scaledROImask = scaled_wsi(wsi_path,mask_filename,scale)//255


# Scaled patch size
(width, height) = (WSIwidth // scale, WSIheight // scale)
scaledpatchsize=patchsize//scale


# ROImask: Create patch-wise mask, 
# pixcoord: Samples coordinates, 
# imcoord: Original Images coordinates
[ROImask_pix,pixcoord,wsicoord]=grtrpixelmask(scaledROImask,
                                               scaledpatchsize,
                                               scale,
                                               th=th)

plt.imshow(ROImask_pix,cmap='gray')

#%% Prediction

model_path = '/home/usuario/Documentos/GBM/TCGA/'
model_file = 'TL_best_model22102021_ResNet50Exp8.h5'

model=keras.models.load_model(model_path+model_file)

wsicoord=np.array(wsicoord) #convert list to array

WSI = Image.fromarray(wsi.read_block(), 'RGB')

del wsi # Delete WSI (Reduce RAM usage)

#%% Patch-wise classification

prediction_list=[]
patchsize = 224

for i in tqdm(range(np.shape(wsicoord)[0])):
    
    top=wsicoord[i,0]
    left=wsicoord[i,1]
    
    # Extracting patch from original WSI (usgin coord)
    WSIpatch=WSI.crop((left,top,left+patchsize,top+patchsize))
    WSIpatch=WSIpatch.resize((patchsize,patchsize)) # Resizing image
    
    WSI_patch_array=np.uint16(np.array(WSIpatch))
    
    # Patch Normalization from (0,255) to (0,1)
    WSI_patch_array_norm = WSI_patch_array/255
    
    # Expand 
    WSI_patch_array_norm =np.expand_dims(WSI_patch_array_norm, axis=0)

    predict_value=model.predict(WSI_patch_array_norm)
    
    prediction_list.append(np.argmax(predict_value))


#%% Gray-level predicted mask (Patch-wise)

pred_mask_pix=np.zeros((np.shape(ROImask_pix)))

coordpix=np.array(pixcoord)

for ind in range(np.shape(coordpix)[0]):
    rowx,colx = coordpix[ind]        
    pred_mask_pix[rowx,colx]=prediction_list[ind]+1
    
plt.imshow(pred_mask_pix)

# Convierte la máscara de pixeles en una de tamaño original
# Convert patchwise-mask to original size
scale = 8
patchsize=224//scale
# pp1=pixtomask(pred_mask_pix,ROImask_pix,patchsize)


scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))

graylevelmask=np.int16(pred_mask_pix*255/2)

ResizedMask = cv.resize(graylevelmask,(np.shape(scaled_WSI)[0],np.shape(scaled_WSI)[1]),
                       interpolation = cv.INTER_AREA)

filename_predmask='/home/usuario/Documentos/GBM/TCGA/' + wsi_name +'_TLpredmask.png'
cv.imwrite(filename_predmask, ResizedMask)

#%% Gray-level predicted mask (Patch-wise)

def smoothmask(imin):
    
    ResizedMask = cv.resize(imin,(np.shape(pred_mask_pix)[1]*4,np.shape(pred_mask_pix)[0]*4),
                        interpolation = cv.INTER_AREA)
    BlurredMask = cv.GaussianBlur(ResizedMask, (5,5),8)
    ModifiedMask = np.uint16(BlurredMask>0.5)
    return ModifiedMask
    
ImClassA=smoothmask(np.int16(pred_mask_pix==1))
ImClassB=smoothmask(np.int16(pred_mask_pix==2))


smoothM=np.zeros((np.shape(ImClassA)[0],np.shape(ImClassA)[1]))
smoothM[ImClassA==1]=1
smoothM[ImClassB==1]=2
plt.imshow(smoothM)

ResizedMask = cv.resize(smoothM,(np.shape(scaled_WSI)[0],np.shape(scaled_WSI)[1]),
                       interpolation = cv.INTER_AREA)

ResizedMask = np.round(ResizedMask)

plt.imshow(ResizedMask,cmap='gray')
plt.axis('off')
ResizedMask2=np.int16(ResizedMask*255/2)

# Save Smooth mask
filename_predsmoothmask='/home/usuario/Documentos/GBM/TCGA/' + wsi_name +'_smoothpredmask.png'
cv.imwrite(filename_predsmoothmask, ResizedMask2)
