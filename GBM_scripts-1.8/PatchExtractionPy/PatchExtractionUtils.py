# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 07:47:43 2021

@author: Andres
"""

from matplotlib import pyplot as plt

import math
import cv2 as cv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def scaled_wsi(path,filename,scale):
    WSI = Image.open(path + filename)

    # Reducing image (Scaled Image)
    (width, height) = (WSI.width // scale, WSI.height // scale)
    #print(str(WSI.width))
    #print(str(WSI.height))
    # Scaled WSI and Image Segmentation
    scaled_WSI = WSI.resize((math.floor(width), math.floor(height)))
    
    scaled_WSI_array = np.array(scaled_WSI)

    return scaled_WSI_array

def wsiregion(WSI_SG,ch1,ch2):
    WSIregionsegm = (WSI_SG[:,:,0]==ch1)*(WSI_SG[:,:,1]==ch2)
    return WSIregionsegm


def grtrpixelmask(img,patchsize,stride,scale,th):

    coord_grtr=[]
    coord_grtr_pix=[]
    
    # stride=112
    # patchsize=224    
    # th=0.95
    
    scaledstride=stride//scale
    scaledpatchsize=patchsize//scale
    
    col=int(np.shape(img)[1]/(scaledstride))
    row=int(np.shape(img)[0]/(scaledpatchsize))
    
    grtr_mask_pix=np.zeros((row,col))

    area_th=scaledpatchsize**2*th
    
    coord=[]
    
    for row_ind in range(row):
        for col_ind in range(col):
            
            a=scaledpatchsize*row_ind
            b=scaledpatchsize*row_ind+scaledpatchsize
            c=scaledstride*col_ind
            d=scaledstride*col_ind+scaledpatchsize
            
            patch_bw = img[a:b,c:d]   
            patch_area=int(np.sum(patch_bw))
            #print(patch_area)
            condition=np.shape(patch_bw)[0]==np.shape(patch_bw)[1]
            
            if condition==True:
                if patch_area>int(area_th):
                #plt.figure()
                #plt.imshow(patch_bw)
                    patch_area=int(np.sum(patch_bw))
                    grtr_mask_pix[row_ind][col_ind]=1
                    coord_grtr.append([row_ind*scaledpatchsize*scale,
                                col_ind*scaledstride*scale])
        #plt.imshow(grtr_mask_pix)
            
    return grtr_mask_pix, coord_grtr


def savepatches(WSI,patchsize,filename,region,coord_grtr,destpath):
    coord_array=np.array(coord_grtr)

    for i in range(len(coord_array)):
        
        top=coord_array[i,0]
        left=coord_array[i,1]
        
        # Extracting patch from original WSI
        
        # im1 = im.crop((left, top, right, bottom))
        WSIpatch=WSI.crop((left,top,left+patchsize,top+patchsize))
        WSIpatch_array=np.array(WSIpatch)
        
        
        patchname = filename[:-4]+'_'+ str(i).zfill(4) + '_'+ region +'.jpg'
        #destpath = 'C:/Users/Andres/Desktop/destino/'
    
        WSIpatch.save(destpath + patchname,format='')
    
    numpatches=len(coord_array)

    return numpatches

    
def pixtomask(imgpix,CTr,ws):
    
    resizedpiximg=np.zeros((np.shape(CTr)))
    col=np.shape(imgpix)[1]
    row=np.shape(imgpix)[0]
    
    for col_ind in range(int(col)):
        for row_ind in range(int(row)):
            resizedpiximg[ws*row_ind:ws*row_ind+ws,
               ws*col_ind:ws*col_ind+ws]=int(imgpix[row_ind][col_ind])
    return resizedpiximg
    



