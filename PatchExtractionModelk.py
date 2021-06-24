# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:38:43 2021

@author: Andres
"""
import math

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from PatchExtractionUtils import scaled_wsi, wsiregion, grtrpixelmask, savepatches

class PatchExtrectionTool():

    def __init__(self,region,ch1,ch2,patchsize,stride,scale,th):
        
        self.ch1=ch1
        self.ch2=ch2
        self.patchsize=patchsize
        self.stride=stride
        self.scale=scale
        self.th=th
        self.region=region
        

    def getWSIregion(self,WSISG_path,filename):
        
        scaled_WSI_SG = scaled_wsi(WSISG_path,'SG_'+filename,scale)
          
        WSI_SG_region=wsiregion(scaled_WSI_SG,ch1,ch2)
        [grtr_mask_pix, coord_grtr]=grtrpixelmask(WSI_SG_region,
                                      patchsize,
                                      stride,
                                      scale,
                                      th)
        
        return grtr_mask_pix,coord_grtr
    
    def getsavepatch(self,WSI_path,filename,patchsize,region,coord_grtr,destpath):
        
        WSI = Image.open(WSI_path + filename)
        num_patches=savepatches(WSI,patchsize,filename,region,coord_grtr,destpath)

        return num_patches

    def run_evaluation(self):
        pass

    def run_training(self):
        pass
    
#%%
mainpath = '/Users/Andres/Downloads/'
destpath='/Users/Andres/Desktop/destino1/'

WSI_path = mainpath + 'WSI/test/'
WSISG_path= mainpath + 'SG/test/'

listfiles = os.listdir(WSI_path)

ind=0

filename=listfiles[ind]

region='CT'
ch1=5
ch2=208
patchsize=448
stride=448
scale=4
th=0.51


mdl = PatchExtrectionTool(region,
                          ch1,
                          ch2,
                          patchsize,
                          stride,
                          scale,
                          th)

[grtr_mask_pix,coord_grtr]=mdl.getWSIregion(WSISG_path,filename)


numero=mdl.getsavepatch(WSI_path,filename,patchsize,region,coord_grtr,destpath)