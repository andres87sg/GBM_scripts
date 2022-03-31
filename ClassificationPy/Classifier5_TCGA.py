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

#%%

from ClassifierUtils import scaled_wsi, wsiregion, grtrpixelmask
from ClassifierUtils import pixtomask


#%%

import slideio
import cv2 as cv
import matplotlib.pyplot as plt


image_path = 'C:/Users/Andres/Desktop/GBM_Project/TCGA_WSI/TCGA-15-0742.svs'
path_mask='C:/Users/Andres/Desktop/GBM_Project/TCGA_WSI/WSI_mask/TCGA-15-0742/'
filename_mask = 'TCGA-15-0742_mask.png'


slide = slideio.open_slide(image_path,"SVS")
wsi = slide.get_scene(0)
magnification = wsi.magnification

block = wsi.read_block()
mask = cv.imread(path_mask)

#%%

scale=2

scaled_WSI_SG = scaled_wsi(path_mask,filename_mask,scale)//255

WSI = Image.fromarray(block, 'RGB')

(width, height) = (WSI.width // scale, WSI.height // scale)

# Scaled WSI and Image Segmentation
scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))

# scaled_WSI_array = np.array(scaled_WSI)


#%%

th=0.7

patchsize = 224

scaledpatchsize=patchsize//scale



CTr=np.uint16(scaled_WSI_SG==1)

(imheigth,imwidth)=np.shape(NEr)

grtr_mask=np.zeros((imheigth,imwidth))

[CTr_pix,CTcoordpix,CTcoord]=grtrpixelmask(CTr,
                                            scaledpatchsize,
                                            scale,
                                            th=th)

grtr_mask_pix = CTr_pix

plt.imshow(grtr_mask_pix,cmap='gray')

# plt.imshow(NEr_pix)

#%% Prediction

import tensorflow.keras as keras

#keras.models.load_model('')
dir='C:/Users/Andres/Desktop/GBM_Project/Experiments/CNN_Models/Model_CRvsNE.h5'
#dir='/home/usuario/Descargas/Model_CRvsNE.h5'
model=keras.models.load_model(dir)

#%%
#coord_array=np.array(NEcoord+CTcoord)
coord_array=np.array(CTcoord)

#%%

prediction_list=[]

# WSI = Image.open(path + filename)
WSI = Image.fromarray(block, 'RGB')
#%%

from tqdm import tqdm

for i in tqdm(range(np.shape(coord_array)[0])):
#for i in range(200,2000):
    
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
    
    prediction_list.append(np.argmax(predict_value))


#%%
pred_mask_pix=np.zeros((np.shape(CTr_pix)))

#coordpix=np.array(NEcoordpix+CTcoordpix)
coordpix=np.array(CTcoordpix)

#%%

for ind in range(np.shape(coordpix)[0]):
#    print(ind)
    rowx,colx = coordpix[ind]        
    pred_mask_pix[rowx,colx]=prediction_list[ind]+1

#%%
plt.imshow(pred_mask_pix)

#%%

# Convierte la máscara de pixeles en una de tamaño original
patchsize=224//2

pp1=pixtomask(pred_mask_pix,CTr,patchsize)
pp2=pixtomask(grtr_mask_pix,CTr,patchsize)

zz=np.int16(pp1>0.5)

#%%

import cv2 as cv
from PIL import Image 
import PIL 

im2=np.int16(pred_mask_pix*255/2)

ResizedMask = cv.resize(im2,(np.shape(scaled_WSI)[1],np.shape(scaled_WSI)[0]),
                       interpolation = cv.INTER_AREA)

filename3="C:/Users/Andres/Downloads/prediction_mask_kkk.png"
cv.imwrite(filename3, ResizedMask)




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

ResizedMask = cv.resize(smoothM,(np.shape(scaled_WSI)[1],np.shape(scaled_WSI)[0]),
                       interpolation = cv.INTER_AREA)

ResizedMask = np.round(ResizedMask)

plt.imshow(ResizedMask,cmap='gray')
plt.axis('off')

#%%

import cv2 as cv
from PIL import Image 
import PIL 

im2=np.int16(ResizedMask*255/2)

filename3="C:/Users/Andres/Downloads/prediction_mask_bbb.jpg"
cv.imwrite(filename3, im2)

#%%
# #%%
# im3 = Image.fromarray(im2, 'RGB')

# #%%
# im1 = im3.save("/home/usuario/Descargas/geeks.jpg")

#jj=np.int16(pp1==1)

#ResizedMask = cv.resize(jj,(np.shape(pred_mask_pix)[1]*4,np.shape(pred_mask_pix)[0]*4),
#                       interpolation = cv.INTER_AREA)

#plt.imshow(ResizedMask)

#%%
# ResizedMask = cv.resize(pp1,(np.shape(pp1)[1]//64,np.shape(pp1)[0]//64),
#                        interpolation = cv.INTER_AREA)

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
# result = cv.morphologyEx(ResizedMask, cv.MORPH_CLOSE, kernel)

# plt.imshow(result)




#%%

# dilated = cv.open(pp1, 
#                      cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), 
#                      iterations=2)


#%%





# plt.imshow(BlurredMask)

# imgoutput2=np.int16(imgoutput>0.5)

# ll=np.zeros((np.shape(pp1)[0]//4,np.shape(pp1)[1]//4,3))

# ll[:,:,0]=imgoutput2
# ll[:,:,1]=imgoutput2
# ll[:,:,2]=imgoutput2

#%%

# for i in range (0,10):

#     ll = cv.GaussianBlur(ll, (5,5), 10)

# plt.imshow(ll)






#a=0
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




