clc;
close all;
clear all;

% Stain Normalization

% Clear all previous data


% Read Image
ImSample = importdata('C:\Users\Andres\Desktop\Otro4.jpg');    
im1 = imresize(ImSample,1/2);
im2 = imresize(ImSample,1/4);
im3 = imresize(ImSample,1/8);

%%
% Normalized image (Function faststain)
[ DCh, H, E, Bg, M ] = Deconvolve(im1, [], 0 );
figure, imshow(E,[]);

%ImOut = E;
% Save image
%imwrite(ImOut,[destpath,filename]);
%disp(numsample)
    
