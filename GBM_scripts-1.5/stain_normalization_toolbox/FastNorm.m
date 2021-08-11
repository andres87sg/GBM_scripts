clc;
clear;
close all;


ref = imread('C:\Users\Andres\Desktop\GBM_Project\Experiments\GBM_scripts\stain_normalization_toolbox\Ref2.png');

im1=imread('C:\Users\Andres\Desktop\PatchesGBM\train11\HB\W1-1-2-Z.1.01_26_HB.jpg');

a=0;

ref2 = imresize(ref,[224,224]);

out=zeros(224,224,3);
%out=size(im1);
% for zz=1:3
%     out(:,:,zz)=imhistmatch(im1(:,:,zz),ref2(:,:,zz));
% end

io=im1;

back=rgb2gray(io)>200;
idx=find(back);
% out=size(io);
for zz=1:3
    ioc=io(:,:,zz);
    refc=ref(:,:,zz);
    ioutt=imhistmatch(ioc(idx),refc(idx));
    ioc(idx)=ioutt;
    io(:,:,zz)=ioc;
end

figure, 
subplot(1,3,1);
imshow(im1,[]);
subplot(1,3,2);
imshow(ref2,[]);
subplot(1,3,3);
imshow(io,[]);




