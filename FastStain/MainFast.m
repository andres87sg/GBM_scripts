
clc;
clear;
close all;

addpath('.\Sample\')

% Reference Image
ImRef= importdata('.\Sample\ref_CT.jpg');
ImRef = imresize(ImRef,[224 224]);


for dir=1:6

    switch dir 

        case 1
            
            path='C:\Users\Andres\Desktop\SetHBCT448x448\train\CT\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\train\CT\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin train CT')

        case 2
            
            path='C:\Users\Andres\Desktop\SetHBCT448x448\train\HB\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\train\HB\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin train HB')

        case 3
            
            path='C:\Users\Andres\Desktop\SetHBCT448x448\val\CT\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\val\CT\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin valid CT')

        case 4
            
            path='C:\Users\Andres\Desktop\SetHBCT448x448\val\HB\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\val\HB\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin valid HB')
            
        case 5
            
            path='C:\Users\Andres\Desktop\SetHBCT448x448\test\CT\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\test\CT\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin test CT')
        
        case 6
            
            path='C:\Users\Andres\Desktop\SetHBCT448x448\test\HB\';
            destpath='C:\Users\Andres\Desktop\SetHBCT448x448_Stain\test\HB\';
            FastStainFunction(path,destpath,ImRef)
            disp('Fin test CT')


    end
    

end

disp('The process has ended')
