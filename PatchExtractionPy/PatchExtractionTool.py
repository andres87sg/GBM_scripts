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
import pandas as pd
Image.MAX_IMAGE_PIXELS = None

from PatchExtractionUtils import scaled_wsi, wsiregion, grtrpixelmask, savepatches

class PatchExtractionTool():

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
        
        numpatches=np.shape(coord_grtr)[0]
        
        return None,coord_grtr,numpatches
        #return grtr_mask_pix,coord_grtr,numpatches
    
    def getsavepatch(self,WSI_path,filename,patchsize,region,coord_grtr,destpath):
        
        WSI = Image.open(WSI_path + filename)
        savepatches(WSI,patchsize,filename,region,coord_grtr,destpath)

        return None
    
    def getpatchestable(self,listopenfiles,patcheslist,regionname):

        filenametable=pd.DataFrame({'filename':listopenfiles})
        patcheslisttable=np.concatenate(patcheslist, axis=0 )
        
        
        df1=pd.DataFrame()
        for i in range(len(regionname)):
            df1[regionname[i]] = np.int16(patcheslisttable[:,i])
            #df1[regionname[1]] = np.int16(patcheslisttable[:,1])
        
        table=pd.concat([filenametable,df1], axis=1)        


        
        return table
        

    def run_evaluation(self):
        pass

    def run_training(self):
        pass
    

# ExtrationParams = 'PatchExtParams.xlsx'

# df = pd.read_excel(ExtrationParams, sheet_name='CurrentExp')

# # Ruta principal y ruta de destino
# mainpath = '/Users/Andres/Downloads/'
# destpath='/Users/Andres/Desktop/destino3/'

# WSI_path = mainpath + 'WSI/train/'
# WSISG_path= mainpath + 'SG/train/'

# listfiles = os.listdir(WSI_path)

# numclasses=df.shape[0]
# patcheslist=[]
# listopenfiles=[]
# len(listfiles)

# for indcase in range(len(listfiles)):
    
#     print('Caso '+ str(indcase+1) +' de ' + str(len(listfiles)))
    
#     filename = listfiles[indcase]    
#     numpatchwsi = np.zeros((1,numclasses))   
#     regionname=[]
    
#     for i in range(numclasses):
#         region=df['region'][i]
#         ch1=df['ch1'][i]
#         ch2=df['ch2'][i]
#         patchsize=df['patchsize'][i]
#         stride=df['stride'][i]
#         scale=df['scale'][i]
#         th=df['th'][i]
    
#         PatchTool = PatchExtractionTool(region,
#                                   ch1,
#                                   ch2,
#                                   patchsize,
#                                   stride,
#                                   scale,
#                                   th)
        
#         [a,coord_grtr,numpatches]=PatchTool.getWSIregion(WSISG_path,filename)
        
#         patchfolder = destpath + region + '/'
        
#         # PatchTool.getsavepatch(WSI_path,
#         #                        filename,
#         #                        patchsize,
#         #                        region,
#         #                        coord_grtr,
#         #                        patchfolder)

#         regionname.append(region)
#         numpatchwsi[0,i]=numpatches
    

#     listopenfiles.append(filename)
#     patcheslist.append(numpatchwsi)


# table=PatchTool.getpatchestable(listopenfiles,patcheslist,regionname)

# print("The process has ended")