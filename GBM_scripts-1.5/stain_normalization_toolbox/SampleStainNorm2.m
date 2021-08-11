% Stain Normalization

% Clear all previous data
clc, 
clear, 
close all;


addpath(path);
mex colour_deconvolution.c;


%%
Source = 'sample100.jpg';
ScrImg=imread(Source);





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

imwrite(NormDM,'esto.jpg')


figure, 
subplot(1,3,1);
imshow(ScrImg,[]);
title('Source');
subplot(1,3,2);
imshow(RefImg,[]);
title('Reference');
subplot(1,3,3);
imshow(NormDM,[]);
title('Normalization');