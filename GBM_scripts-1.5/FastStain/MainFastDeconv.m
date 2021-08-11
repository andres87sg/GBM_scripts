
clc;
clear;
close all;

%addpath('.\Sample\')

% Reference Image
path = './Deconvolve/';
addpath(path);
mex ./Deconvolve/colour_deconvolution.c;


for dir=1:6

    switch dir 

        case 1
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/train/CT/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_decon/train/CT/';
            FastDeconvolution(path,destpath)
            disp('Fin train CT')

        case 2
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/train/HB/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_decon/train/HB/';
            FastDeconvolution(path,destpath)
            disp('Fin train HB')

        case 3
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/val/CT/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_decon/val/CT/';
            FastDeconvolution(path,destpath)
            disp('Fin valid CT')

        case 4
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/val/HB/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_decon/val/HB/';
            FastDeconvolution(path,destpath)
            disp('Fin valid HB')
            
        case 5
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/test/CT/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_decon/test/CT/';
            FastDeconvolution(path,destpath)
            disp('Fin test CT')
        
        case 6
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/test/HB/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_decon/test/HB/';
            FastDeconvolution(path,destpath)
            disp('Fin test CT')


    end
    

end

disp('The process has ended')

