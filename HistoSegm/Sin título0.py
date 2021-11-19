#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:48:46 2021

@author: usuario
# """
# path = '/home/usuario/Documentos/LungInf/LungInfDataset/MedSegData/Mask/Mask/'
# destpath  = '/home/usuario/Documentos/LungInf/LungInfDataset/MedSegData/Mask2/Mask2/'

path = '/home/usuario/Documentos/LungInf/LungInfDataset/Validation/Mask/Mask/'
destpath = '/home/usuario/Documentos/LungInf/LungInfDataset/Validation/Mask3/Mask3/'

mask_listfiles = sorted(os.listdir(path))

#for i in range(1,20):
for i in range(len(mask_listfiles)):


  # List of files
  mask_im_name = mask_listfiles[i]
  mask_array=cv2.imread(path+mask_im_name)   # Mask image
  
  colormask = np.zeros((512,512,3))
  
  colormask[:,:,0]=mask_array[:,:,0]==0 
  colormask[:,:,1]=mask_array[:,:,0]==127
  colormask[:,:,2]=mask_array[:,:,0]==255
  
  for ch in range(3):
      colormask[:,:,ch][colormask[:,:,ch]==1]=np.int16(255)

  
  # plt.show()
  # plt.imshow(colormask)
  
  cv2.imwrite(destpath+mask_im_name,colormask)
