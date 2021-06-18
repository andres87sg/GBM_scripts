# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:30:36 2020

@author: Andres
"""
#%%

#import PIL
import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from ClassifierUtils import scaled_wsi, wsiregion, grtrpixelmask
from ClassifierUtils import pixtomask

path='C:/Users/Andres/Downloads/WSI/test/'
path_SG='C:/Users/Andres/Downloads/SG/test/'

filename='W48-1-1-M.04.jpg'

scale=4
th=0.5
patchsize=224



scaled_WSI = scaled_wsi(path,filename,scale)
scaled_WSI_SG = scaled_wsi(path_SG,'SG_'+filename,scale)

# Selecting 
NEr=wsiregion(scaled_WSI_SG,ch1=5,ch2=5)
CTr=wsiregion(scaled_WSI_SG,ch1=5,ch2=208)

# Create groundtruth mask
(imheigth,imwidth,x)=np.shape(scaled_WSI)
grtr_mask=np.zeros((imheigth,imwidth))

grtr_mask[NEr==1]=1
grtr_mask[CTr==1]=2


[NEr_pix,NEcoordpix,NEcoord]=grtrpixelmask(NEr,
                                            patchsize,
                                            scale,
                                            th=th)
        
[CTr_pix,CTcoordpix,CTcoord]=grtrpixelmask(CTr,
                                            patchsize,
                                            scale,
                                            th=th)

grtr_mask_pix=np.zeros((np.shape(CTr_pix)))
grtr_mask_pix[NEr_pix==1]=1
grtr_mask_pix[CTr_pix==1]=2

plt.imshow(grtr_mask_pix,cmap='gray')

#%% Prediction

import tensorflow.keras as keras

dir='C:/Users/Andres/Desktop/GBM_Project/Experiments/CNN_Models/Model_CRvsNE.h5'
model=keras.models.load_model(dir)

#%%
coord_array=np.array(NEcoord+CTcoord)

prediction_list=[]

WSI = Image.open(path + filename)

from tqdm import tqdm

for i in tqdm(range(np.shape(coord_array)[0])):
    
    top=coord_array[i,0]
    left=coord_array[i,1]
    
    # Extracting patch from original WSI
    
    # im1 = im.crop((left, top, right, bottom))
    WSIpatch=WSI.crop((left,top,left+patchsize,top+patchsize))
    WSI_patch_array=np.array(WSIpatch)
    WSI_patch_array_norm = WSI_patch_array/255 
    
    # Expand 
    WSI_patch_array_norm=np.expand_dims(WSI_patch_array_norm, axis=0)
    predict_value=model.predict(WSI_patch_array_norm)
    prediction_list.append(round(predict_value[0,1]))

#%%
pred_mask_pix=np.zeros((np.shape(CTr_pix)))

coordpix=np.array(NEcoordpix+CTcoordpix)

for ind in range(len(coordpix)):
    rowx,colx = coordpix[ind]        
    pred_mask_pix[rowx,colx]=np.array(prediction_list[ind])+1


pp1=pixtomask(pred_mask_pix,CTr,patchsize)
pp2=pixtomask(grtr_mask_pix,CTr,patchsize)

a=0
# #%%

# resized=np.int16(resized)
# ll=np.zeros((9024//8,7520//8))
# ll[resized==1]=1

# ll1=np.zeros((9024//8,7520//8,3))
# ll1[:,:,0]=0
# ll1[:,:,1]=ll*255
# ll1[:,:,2]=0


# #%%

# added_image = cv.addWeighted(ll1,1,wsiscal/255,0.6,0)
# plt.imshow(added_image)
# plt.axis('off')

# #%%
# from skimage import io, color

# overlapimg=color.label2rgb(resized,wsiscal[:,:,:]/255,
#                           colors=[(0,0,0),(0,1,0)],
#                           alpha=0.4, bg_label=0, bg_color=None)

# plt.imshow(overlapimg)




