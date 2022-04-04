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



import slideio
import cv2 as cv
import matplotlib.pyplot as plt



wsi_name = 'TCGA-15-0742'
wsi_path = '/home/usuario/Documentos/GBM/TCGA/'
mask_filename = wsi_name + '_mask.png'
wsi_filename = wsi_name + '.svs'
# #%%

# image_path = '/home/usuario/Documentos/GBM/TCGA/TCGA-15-0742.svs'
# path_mask = '/home/usuario/Documentos/GBM/TCGA/'
# filename_mask = 'TCGA-15-0742_mask.png'


#%%

# image_path = 'C:/Users/Andres/Desktop/GBM_Project/TCGA_WSI/TCGA-15-0742.svs'
# path_mask='C:/Users/Andres/Desktop/GBM_Project/TCGA_WSI/WSI_mask/TCGA-15-0742/'
# filename_mask = 'TCGA-15-0742_mask.png'TCGA-02-0336
a=0

slide = slideio.open_slide(wsi_path + wsi_filename,"SVS")
wsi = slide.get_scene(0)
magnification = wsi.magnification
WSIheight,WSIwidth =wsi.size
# block = wsi.read_block()
mask = cv.imread(wsi_path + mask_filename)

#%%

scale=2

scaled_WSI_SG = scaled_wsi(wsi_path,mask_filename,scale)//255

# WSI = Image.fromarray(block, 'RGB')

(width, height) = (WSIwidth // scale, WSIheight // scale)

# Scaled WSI and Image Segmentation
# scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))

# scaled_WSI_array = np.array(scaled_WSI)


#%%

th=0.5

patchsize = 224

scaledpatchsize=patchsize//scale



CTr=np.uint16(scaled_WSI_SG==1)

(imheigth,imwidth)=np.shape(CTr)

grtr_mask=np.zeros((imheigth,imwidth))

[CTr_pix,CTcoordpix,CTcoord]=grtrpixelmask(CTr,
                                            scaledpatchsize,
                                            scale,
                                            th=th)

grtr_mask_pix = CTr_pix

plt.imshow(grtr_mask_pix,cmap='gray')

# plt.imshow(NEr_pix)

#%% Prediction
#a
import tensorflow.keras as keras

#keras.models.load_model('')
#dir='C:/Users/Andres/Desktop/GBM_Project/Experiments/CNN_Models/Model_CRvsNE.h5
model_path = '/home/usuario/Documentos/GBM/TCGA/'
model_file = 'best_model19102021_Eff3Exp7.h5'

model=keras.models.load_model(model_path+model_file)

#%%
coord_array=np.array(CTcoord)

prediction_list=[]

# WSI = Image.open(path + filename)
block = wsi.read_block()
WSI = Image.fromarray(block, 'RGB')

from tqdm import tqdm

for i in tqdm(range(np.shape(coord_array)[0])):
# for i in range(251,252):
    
    top=coord_array[i,0]
    left=coord_array[i,1]
    
    # Extracting patch from original WSI
    
    # im1 = im.crop((left, top, right, bottom))
    WSIpatch=WSI.crop((left,top,left+patchsize,top+patchsize))
    # WSIpatch=WSIpatch.resize((112,112)) # Resizing image
    WSI_patch_array=np.array(WSIpatch)
    WSI_patch_array_norm = WSI_patch_array/255
    
    # Expand 
    WSI_patch_array_norm=np.expand_dims(WSI_patch_array_norm, axis=0)
    predict_value=model.predict(WSI_patch_array_norm)
    
    prediction_list.append(np.argmax(predict_value))

#%%
# ResizedMask = cv.resize(WSIpatch,(112,112),interpolation = cv.INTER_AREA)


# ResizedMask = cv.resize(WSIpatch,(112,112))
# 

#%%
pred_mask_pix=np.zeros((np.shape(CTr_pix)))

#coordpix=np.array(NEcoordpix+CTcoordpix)
coordpix=np.array(CTcoordpix)

#%%

for ind in range(np.shape(coordpix)[0]):
#    print(ind)
    rowx,colx = coordpix[ind]        
    pred_mask_pix[rowx,colx]=prediction_list[ind]+1
    
plt.imshow(pred_mask_pix)

#%%

# Convierte la máscara de pixeles en una de tamaño original
patchsize=224//scale
pp1=pixtomask(pred_mask_pix,CTr,patchsize)
# pp2=pixtomask(grtr_mask_pix,CTr,patchsize)

# zz=np.int16(pp1>0.5)
#%%

scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))

#%%

import cv2 as cv
from PIL import Image 
import PIL 

im2=np.int16(pred_mask_pix*255/2)

ResizedMask = cv.resize(im2,(np.shape(scaled_WSI)[0],np.shape(scaled_WSI)[1]),
                       interpolation = cv.INTER_AREA)

filename3='/home/usuario/Documentos/GBM/TCGA/' + wsi_name +'_predmask.png'
cv.imwrite(filename3, ResizedMask)


# imin = np.int16(pred_mask_pix==2)

#%%

# def smoothmask(imin):

#     for zz in range(1):
        
#         ResizedMask = cv.resize(imin,(np.shape(pred_mask_pix)[1]*2,np.shape(pred_mask_pix)[0]*2),
#                            interpolation = cv.INTER_AREA)
#         BlurredMask = cv.GaussianBlur(ResizedMask, (5,5),8*zz)
#         ModifiedMask = np.uint16(BlurredMask>0.5)
        
#         imin = ModifiedMask 
    
#     return imin

# plt.imshow(imin)

#%%

def smoothmask(imin):
    
    ResizedMask = cv.resize(imin,(np.shape(pred_mask_pix)[1]*4,np.shape(pred_mask_pix)[0]*4),
                        interpolation = cv.INTER_AREA)
    BlurredMask = cv.GaussianBlur(ResizedMask, (5,5),8)
    ModifiedMask = np.uint16(BlurredMask>0.5)
    return ModifiedMask
    
imA=smoothmask(np.int16(pred_mask_pix==1))
imB=smoothmask(np.int16(pred_mask_pix==2))


smoothM=np.zeros((np.shape(imA)[0],np.shape(imA)[1]))
smoothM[imA==1]=1
smoothM[imB==1]=2

plt.imshow(smoothM)

#%%

ResizedMask = cv.resize(smoothM,(np.shape(scaled_WSI)[0],np.shape(scaled_WSI)[1]),
                       interpolation = cv.INTER_AREA)

ResizedMask = np.round(ResizedMask)

plt.imshow(ResizedMask,cmap='gray')
plt.axis('off')


im2=np.int16(ResizedMask*255/2)

filename3='/home/usuario/Documentos/GBM/TCGA/' + wsi_name +'_smoothpredmask.png'
cv.imwrite(filename3, im2)
