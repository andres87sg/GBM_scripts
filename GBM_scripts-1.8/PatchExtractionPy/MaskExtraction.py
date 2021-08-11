# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:30:36 2020

@author: Andres
"""
#%%

#import PIL
import math

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from PatchExtractionUtils import scaled_wsi, wsiregion, grtrpixelmask, savepatches

#%%

mainpath = '/Users/Andres/Desktop/HB_Unet/896x896/train/SG/'

# WSI_path = mainpath+'WSI/test/'
# WSISG_path= mainpath +'SG/test/'

listfiles = os.listdir(mainpath)

#%%
import cv2 as cv
from scipy import misc
from scipy.ndimage import gaussian_filter

NE=(5,5)
HB=(255,101)
ch1=HB[0]
ch2=HB[1]

imgmask = Image.open(mainpath+listfiles[2])
imgmask=np.array(imgmask)

mask = (imgmask[:,:,0]==ch1)*(imgmask[:,:,1]==ch2)
plt.imshow(mask,cmap='gray')

 WSIpatch.save(destpath + patchname)


#%%
from tqdm import tqdm

for ind in range(0,1):
#for ind in tqdm(range(0,1)):
#for ind in tqdm(range(len(listfiles))):
    #filename='W48-1-1-M.02'+'.jpg'
    
    filename=listfiles[ind]
    WSI = Image.open(WSI_path + filename)
    WSI_SG = Image.open(WSISG_path + 'SG_' + filename)
    
    patchsize=448
    stride=448
    scale=4
    th=0.51
    
    scaled_WSI = scaled_wsi(WSI_path,filename,scale)
    scaled_WSI_SG = scaled_wsi(WSISG_path,'SG_'+filename,scale)
      
    WSI_SG_region=wsiregion(scaled_WSI_SG,ch1=HB[0],ch2=HB[1])
    [grtr_mask_pix, coord_grtr]=grtrpixelmask(WSI_SG_region,
                                  patchsize,
                                  stride,
                                  scale,
                                  th)
    
    destpath='C:/Users/Andres/Desktop/destino1/'
    savepatches(WSI,patchsize,filename,coord_grtr,destpath)
    
    destpath2='C:/Users/Andres/Desktop/destino2/'
    savepatches(WSI_SG,patchsize,filename,coord_grtr,destpath2)
