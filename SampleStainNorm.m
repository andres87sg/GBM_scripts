% Stain Normalization

% Clear all previous data
clc, 
clear, 
close all;


addpath('.\stain_normalization_toolbox\');
mex .\stain_normalization_toolbox\colour_deconvolution.c;

%% Load input & reference image
Source = 'sample100.jpg';
img_src=imread(Source);
ref=imread('Ref3.jpg');


if exist('normalised/', 'dir') == 7
    rmdir('normalised', 's');
end

if exist('normalised/', 'dir')==0
    mkdir('normalised/');
end

dos(['ColourNormalisation.exe BimodalDeconvRVM filename.txt', ...
    ' Ref.png HE.colourmodel']);
% pause(4);
NormDM = imread(['.\stain_normalization_toolbox\normalised\', Source]);

figure, 
subplot(1,3,1);
imshow(img_src,[]);
title('Source');
subplot(1,3,2);
imshow(ref,[]);
title('Reference');
subplot(1,3,3);
imshow(NormDM,[]);
title('Normalization');