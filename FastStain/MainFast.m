
clc;
clear;
close all;

addpath('.\Sample\')

% Reference Image
ImRef= importdata('.\Sample\ref5.jpg');
ImRef = imresize(ImRef,[224 224]);


for dir=5:6

    switch dir 

        case 1
            
            path='C:\Users\Andres\Desktop\PatchesGBM\train11\CT\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\train12\CTnorm3\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin train CT')

        case 2
            
            path='C:\Users\Andres\Desktop\PatchesGBM\train11\HB\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\train12\HBnorm3\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin train HB')

        case 3
            
            path='C:\Users\Andres\Desktop\PatchesGBM\valid11\CT\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\valid12\CTnorm3\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin valid CT')

        case 4
            
            path='C:\Users\Andres\Desktop\PatchesGBM\valid11\HB\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\valid12\HBnorm3\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin valid HB')
            
        case 5
            
            path='C:\Users\Andres\Desktop\PatchesGBM\test11\CT\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\test12\CTnorm3\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin test CT')
        
        case 6
            
            path='C:\Users\Andres\Desktop\PatchesGBM\test11\HB\';
            destpath='C:\Users\Andres\Desktop\PatchesGBM\test12\HBnorm3\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin test CT')


    end
    

end

disp('The process has ended')

