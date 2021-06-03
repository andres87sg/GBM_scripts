
clc;
clear;
close all;

addpath('.\Sample\')

% Reference Image
path = '.\Deconvolve\';
addpath(path);
mex .\Deconvolve\colour_deconvolution.c;


for dir=2:2

    switch dir 

        case 1
            
            path='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448\train\CT\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448_Deco\train\CT\';
            FastDeconvolution(path,destpath)
            disp('Fin train CT')

        case 2
            
            path='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448\train\HB\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448_Deco\train\HB\';
            FastDeconvolution(path,destpath)
            disp('Fin train HB')

        case 3
            
            path='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448\val\CT\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\val\CT\';
            FastDeconvolution(path,destpath,ImRef)
            disp('Fin valid CT')

        case 4
            
            path='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448\val\HB\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\val\HB\';
            FastDeconvolution(path,destpath,ImRef)
            disp('Fin valid HB')
            
        case 5
            
            path='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448\test\CT\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\test\CT\';
            FastDeconvolution(path,destpath,ImRef)
            disp('Fin test CT')
        
        case 6
            
            path='C:\Users\Andres\Desktop\PatchesGBM\SetHBCT448x448\test\HB\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\test\HB\';
            FastDeconvolution(path,destpath,ImRef)
            disp('Fin test CT')


    end
    

end

disp('The process has ended')

