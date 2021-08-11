%{

Author: Andres Sandino
Created: 07/05/2021

Source: 
On Stain normalization in deep learning
http://www.andrewjanowczyk.com/on-stain-normalization-in-deep-learning/

%}


clc;
clear;
close all;

addpath('.\Sample\')

ImRef= importdata('.\Sample\ref5.jpg');

path='C:\Users\Andres\Desktop\PatchesGBM\train11\CT\';
destpath='C:\Users\Andres\Desktop\PatchesGBM\train12\CTnorm3\';
readfolder=dir(strcat(path,'*.jpg'));

% Reference Image
ImRef = imresize(ImRef,[224 224]);

%for numsample = 1:size(read_folder,1)
for numsample = 1:1000
    
    % Read Image filename
    filename = readfolder(numsample).name;     
    % Read Image
    ImSample = importdata([path,filename]);    
    % Normalized image
    ImOut = faststain(ImSample,ImRef);
    % Save image
    imwrite(ImOut,[destpath,filename]);
    %disp(numsample)
    
end

% figure,
% subplot(1,3,1);
% imshow(ImSample,[]);
% title('Original');
% subplot(1,3,2);
% imshow(ImRef,[]);
% title('Reference');
% subplot(1,3,3);
% imshow(ImOut,[]);
% title('Reference');



%% Fast stain (quick and dirty)

function ImOut=faststain(ImIn,ImRef)

    out=zeros(size(ImRef));
    
    for zz=1:3
        out(:,:,zz)=imhistmatch(ImIn(:,:,zz),ImRef(:,:,zz));
    end

    back=rgb2gray(ImIn)>100;
    idx=find(back);
    out=zeros(size(ImRef));
        for zz=1:3
            ioc=ImIn(:,:,zz);
            refc=ImRef(:,:,zz);
            ioutt=imhistmatch(ioc(idx),refc(idx));
            ioc(idx)=ioutt;
            ImOut(:,:,zz)=ioc;
        end
a=0;
        
end
