# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:27:03 2021

@author: Andres
"""
import cv2 as cv

#%%
path = 'C:/Users/Andres/Desktop/destino5/PC_SG/'
destpath = 'C:/Users/Andres/Desktop/destino5/PC_SG2/'
listfiles = os.listdir(path)
listfiles.sort()

for i in range(len(listfiles)):
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