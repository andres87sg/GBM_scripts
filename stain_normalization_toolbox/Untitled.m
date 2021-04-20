clc; clear; close all;

% Source = 'Source_small.png';
Source = 'sample2.png';
img_src=imread(Source);
ref=imread('Ref5.jpg');

disp('Stain Normalization using Macenko et al Method');

verbose = 1;
[NormMM ] = Macenko(img_src, ref, 255, 0.15, 1, verbose);

%%

verbose = 0;
stain_matrix = EstStainUsingMacenko( img_src );

disp(['Color Deconvolution using Our Implementation with Standard ', ...
    'Stain Matrix']);
[ DCh, H, E, Bg ] = Deconvolve( img_src, stain_matrix', verbose );

% 
% %%
% if exist('normalised/', 'dir')==0
%     mkdir('normalised/');
% end
% 
% dos(['ColourNormalisation.exe BimodalDeconvRVM filename.txt', ...
%     ' Ref.png HE.colourmodel']);
% % pause(4);
% NormDM = imread(['normalised/', Source]);