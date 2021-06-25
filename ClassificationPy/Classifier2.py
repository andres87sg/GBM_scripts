# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:30:36 2020

@author: Andres
"""
#%%

#import PIL
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

path='C:/Users/Andres/Downloads/WSI/test/'
path_SG='C:/Users/Andres/Downloads/SG/test/'

# Read WSI and Segmentation
WSI = Image.open(path + 'W48-1-1-M.04.jpg')
WSI_SG = Image.open(path_SG + 'SG_W48-1-1-M.04.jpg')

#print(WSI.mode)
#print(WSI.size)
#image.show()

#%%

scale = 2

# Reducing image (Scaled Image)
(width, height) = (WSI.width // scale, WSI.height // scale)

# Scaled WSI and Image Segmentation
scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))
scaled_WSI_SG = WSI_SG.resize((math.floor(width), math.floor(height)))

#%%

# Green, blue and purple regions 

scaled_WSI_SG = np.array(scaled_WSI_SG)

im_green = (scaled_WSI_SG[:,:,0]==5)*(scaled_WSI_SG[:,:,1]==208)
im_blue = (scaled_WSI_SG[:,:,0]==33)*(scaled_WSI_SG[:,:,1]==143)
im_purple = (scaled_WSI_SG[:,:,0]==210)*(scaled_WSI_SG[:,:,1]==5)
NEr = (scaled_WSI_SG[:,:,0]==5)*(scaled_WSI_SG[:,:,1]==5)



# Choose segmentation mask

maskCT = im_green # CT mask
mask_nonCT = NEr
#mask_nonCT = (im_blue + im_purple) # Non-CT mask

mask_gt = mask_nonCT + 2*maskCT

mask_bw = np.int16(np.array(mask_gt>=1))
#mask_bw = np.array(mask_gt>=1).astype(np.int)

# Show b
img = plt.imshow(mask_gt,cmap='gray')
plt.title('Mask')
plt.axis('off')


#%%

# mask_bw=img
patchsize = 224
scale = 2
ws = int(patchsize/scale) # Scaled window size


th=0.95
     
col = np.floor(scaled_WSI.width/ws)
row = np.floor(scaled_WSI.height/ws)
    
#%%

# Extract groundtruth mask and patches coordinates

def pixel_mask(img,scale,ws,patchsize,row,col,th):
    
    k=0
    
    area_th=(ws**2)*th
    # Groundtruth patches
    grtr_mask=np.zeros((int(row),int(col)))
    
    coord=[]
    coord2=[]
    
    for col_ind in range(int(col)):
        for row_ind in range(int(row)):
    
            # Patch BW
            patch_bw = img[ws*row_ind:ws*row_ind+ws,ws*col_ind:ws*col_ind+ws]   
            
            # Thresholded image
            #patch_bw_th=np.array(patch_bw>=1).astype(np.int)
            patch_bw_th=np.int16(np.array(patch_bw>0))
            
            # Compute segmented area
            #patch_area=np.sum(patch_bw_th)
            patch_area=sum(sum(patch_bw_th))
            
            
            if patch_area>np.int(area_th):
                #grtr_mask[row_ind][col_ind]=np.median(patch_bw)
                k=k+1
                print(k)
                grtr_mask[row_ind][col_ind]=1
                coord.append([row_ind,col_ind])
                #grtr_mask[row_ind][col_ind]=np.max(patch_bw)
                
                #coord.append([row_ind,col_ind])
                
            else:
                #k=k+1
                #print(k)
                grtr_mask[row_ind][col_ind]=np.median(patch_bw)
                grtr_mask[row_ind][col_ind]=np.max(patch_bw)
    
    return grtr_mask,coord,coord2

#%%

# DEBO CORREGIR ESTOOOOO!!! HAY UN ERROR EN COOORD!! NO SÉ QUE PASA

mask_bw_mini,coord,coord2 = pixel_mask(mask_bw,scale,ws,patchsize,row,col,th)

plt.figure(2)
plt.imshow(mask_bw_mini,cmap='gray')
plt.title('bw_mask_mini')
plt.axis('off')
            

#%%
mask_grtr_mini,coord3,coord4 = pixel_mask(mask_gt,scale,ws,patchsize,row,col,th)
#a1,b1 = reduce_mask(mask_gt,scale,patchsize,row,col,area_th)

#%%

plt.figure(1)
plt.imshow(mask_grtr_mini,cmap='gray')
plt.title('grtr_mask_mini')
plt.axis('off')

plt.figure(2)
plt.imshow(mask_bw_mini,cmap='gray')
plt.title('bw_mask_mini')
plt.axis('off')
            
#%% Esto es una prueba

# coord_array=np.array(coord)

# for i in range(0,29):
    
#     # Reescaling window = ws*scale
#     top=coord_array[i,0]*ws*scale
#     left=coord_array[i,1]*ws*scale
    
#     # Extracting patch from original WSI
    
#     # im1 = im.crop((left, top, right, bottom))
#     WSI_patch_image=WSI.crop((left,top,left+ws*scale,top+ws*scale))
    
#     # Normalized patch
#     WSI_patch_array=np.array(WSI_patch_image)
#     WSI_patch_array_normalized = WSI_patch_array//255 
# #    plt.figure()
# #    plt.imshow(WSI_patch_array)
# #    plt.axis('off')
    
    
#%% Prediction

#import tensorflow as tf
import tensorflow.keras as keras
#import math

dir='C:/Users/Andres/Desktop/Model_CRvsNE.h5'
model=keras.models.load_model(dir)

coord_array=np.array(coord)

prediction_list=[]

for i in range(0,len(coord)):
    
    # Reescaling window = ws*scale
    top=coord_array[i,0]*ws*scale
    left=coord_array[i,1]*ws*scale
    
    # Extracting patch from original WSI
    
    # im1 = im.crop((left, top, right, bottom))
    WSI_patch_image=WSI.crop((left,top,left+ws*scale,top+ws*scale))
    
    # Normalized patch
    WSI_patch_array=np.array(WSI_patch_image)
    WSI_patch_array_norm = WSI_patch_array/255 
    
    # Expand 
    WSI_patch_array_norm=np.expand_dims(WSI_patch_array_norm, axis=0)
    predict_value=model.predict(WSI_patch_array_norm)
    prediction_list.append(round(predict_value[0,1]))
    
#%% Predicted Heatmap

#col = np.floor(scaled_WSI.width/ws)
#row = np.floor(scaled_WSI.height/ws)
col = np.int16(scaled_WSI.width/ws)
row = np.int16(scaled_WSI.height/ws)

# Groundtruth patches
predict_mask_mini=np.zeros((int(row),int(col)))
plt.imshow(predict_mask_mini,cmap='gray')

#%%

for ind in range(0,len(coord)):
    rowx,colx = coord[ind]        
    # prediction_list=1 (Non-CT) , prediction_list=2 (CT)
    #predict_mask_mini[rowx,colx]=round(prediction_list[ind]+1)
    predict_mask_mini[rowx,colx]=np.array(prediction_list[ind])+1



plt.figure()
plt.imshow(predict_mask_mini,cmap='gray')
plt.axis('off')
plt.title('predicted mask')


#%%

import scipy as sc
from scipy import ndimage

im_med = ndimage.median_filter(predict_mask_mini, 3)

plt.figure()
plt.imshow(im_med,cmap='gray')
plt.axis('off')
plt.title('filtered mask', fontsize=15)


