
clc;
clear;
close all;

addpath('./Sample/')
addpath('./deconvolve/')

% Reference Image
ImRef= importdata('./Sample/ref_CT2.jpg');
ImRef = imresize(ImRef,[224 224]);

%[ ~, H, E, Bg, ~ ] = Deconvolve( ImRef, [], 0 );


for dir=1:6

    switch dir 

        case 1
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/train/CT/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_stain/train/CT/';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin train CT')

        case 2
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/train/HB/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_stain/train/HB/';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin train HB')

        case 3
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/val/CT/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_stain/val/CT/';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin valid CT')

        case 4
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/val/HB/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_stain/val/HB/';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin valid HB')
            
        case 5
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/test/CT/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_stain/test/CT/';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin test CT')
        
        case 6
            
            path='/home/usuario/Documentos/GBM/Samples/samples_224x224/test/HB/';
            destpath='/home/usuario/Documentos/GBM/Samples/samples_224x224_stain/test/HB/';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin test CT')


    end
    

end

disp('The process has ended')

