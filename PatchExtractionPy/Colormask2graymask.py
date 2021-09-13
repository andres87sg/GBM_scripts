# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:27:03 2021

@author: Andres
"""
import cv2 as cv
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import join

#%%
path = 'C:/Users/Andres/Desktop/destino 7/PC_SG/'
destpath = 'C:/Users/Andres/Desktop/destino 7/PC_SG2/'
listfiles = listdir(path)
listfiles.sort()

for i in tqdm(range(len(listfiles))):
    filename=listfiles[i]
    
    im1 = Image.open(path + filename)
    im2 = np.array(im1)
    
    
    ch1 = 6
    ch2 = 208
    
    im3 = (im2[:,:,0]==ch1) * (im2[:,:,1]==ch2)
    im4 = np.uint8(im3)*255
    kernel = np.ones((20,20),np.uint8)
    im5 = cv.morphologyEx(im4, cv.MORPH_OPEN, kernel)
    
    cv.imwrite(destpath + filename , im5)