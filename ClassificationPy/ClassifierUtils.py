# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 07:47:43 2021

@author: Andres
"""

import math
import cv2 as cv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def scaled_wsi(path,filename,scale):
    WSI = Image.open(path + filename)
    # scale = 2

    # Reducing image (Scaled Image)
    (width, height) = (WSI.width // scale, WSI.height // scale)

    # Scaled WSI and Image Segmentation
    scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))
    
    scaled_WSI_array = np.array(scaled_WSI)

    return scaled_WSI_array

def wsiregion(WSI_SG,ch1,ch2):
    WSIregionsegm = (WSI_SG[:,:,0]==ch1)*(WSI_SG[:,:,1]==ch2)
    return WSIregionsegm


def grtrpixelmask(img,patchsize,scale,th):
    
    coord_grtr=[]
    coord_grtr_pix=[]

    #ws = patchsize//scale # Scaled window size
    ws=patchsize
    
    col = int(np.shape(img)[1]//patchsize)
    row = int(np.shape(img)[0]//patchsize)
    print(col)
    print(row)
    
    grtr_mask_pix=np.zeros((row,col))
    
    area_th=ws**2*th
    #mask=grtr_mask==1
    
    coord=[]
    k=0
    for col_ind in range(int(col)):
        for row_ind in range(int(row)):
        
            patch_bw = img[ws*row_ind:ws*row_ind+ws,
                           ws*col_ind:ws*col_ind+ws]   
            
            patch_area=int(np.sum(patch_bw))
            
            if patch_area>int(area_th):

                grtr_mask_pix[row_ind][col_ind]=1
                coord_grtr.append([row_ind*ws*scale,col_ind*ws*scale])
                coord_grtr_pix.append([row_ind,col_ind])
                
    
                
    return grtr_mask_pix,coord_grtr_pix,coord_grtr

def pixtomask(imgpix,CTr,ws):
    
    resizedpiximg=np.zeros((np.shape(CTr)))
    col=np.shape(imgpix)[1]
    row=np.shape(imgpix)[0]
    
    for col_ind in range(int(col)):
        for row_ind in range(int(row)):
            resizedpiximg[ws*row_ind:ws*row_ind+ws,
               ws*col_ind:ws*col_ind+ws]=int(imgpix[row_ind][col_ind])
    return resizedpiximg
    



