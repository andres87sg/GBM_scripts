# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 06:41:55 2021

@author: Andres
"""
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

import PatchExtractionTool

#%%
ExtrationParams = 'PatchExtParams.xlsx'

df = pd.read_excel(ExtrationParams, sheet_name='CurrentExp')

#%%

# Ruta principal y ruta de destino
mainpath = '/Users/Andres/Downloads/'
destpath='/Users/Andres/Desktop/destino3/'

WSI_path = mainpath + 'WSI/train/'
WSISG_path= mainpath + 'SG/train/'

listfiles = os.listdir(WSI_path)

numclasses=df.shape[0]
patcheslist=[]
listopenfiles=[]
len(listfiles)

for indcase in range(len(listfiles)):
    
    print('Caso '+ str(indcase+1) +' de ' + str(len(listfiles)))
    
    filename = listfiles[indcase]    
    numpatchwsi = np.zeros((1,numclasses))   
    regionname=[]
    
    for i in range(numclasses):
        region=df['region'][i]
        ch1=df['ch1'][i]
        ch2=df['ch2'][i]
        patchsize=df['patchsize'][i]
        stride=df['stride'][i]
        scale=df['scale'][i]
        th=df['th'][i]
    
        PatchTool = PatchExtractionTool(region,
                                  ch1,
                                  ch2,
                                  patchsize,
                                  stride,
                                  scale,
                                  th)
        
        [a,coord_grtr,numpatches]=PatchTool.getWSIregion(WSISG_path,filename)
        
        patchfolder = destpath + region + '/'
        
        # PatchTool.getsavepatch(WSI_path,
        #                        filename,
        #                        patchsize,
        #                        region,
        #                        coord_grtr,
        #                        patchfolder)

        regionname.append(region)
        numpatchwsi[0,i]=numpatches
    

    listopenfiles.append(filename)
    patcheslist.append(numpatchwsi)


table=PatchTool.getpatchestable(listopenfiles,patcheslist,regionname)

print("The process has ended")