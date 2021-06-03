%%%%%%%%%
% Fecha creación: 17/Ene/2021
% Fecha modificado: 17/Ene/2021
% Abrir archivos WSI y extraer parches cuadrados
% Nota: la imagen original está en im_in=imread(file_name);
%%%%%%%%%%

clc;
clear;
close all;
%% Load classification model

modelfile = 'C:\Users\Andres\Desktop\Model_CRvsNE.h5';
% modelfile = 'C:\Users\Andres\Desktop\GBM_Project\Experiments\CNN_Models\best_modelExp6_23102020.h5';
model = importKerasNetwork(modelfile,'OutputLayerType','classification','Classes',{'0','1'});
% model = importKerasNetwork(modelfile,'OutputLayerType','classification');

%%
path='C:\Users\Andres\Downloads\WSI\test\';

% wsi: Whole Slide Image
wsi=importdata('C:\Users\Andres\Downloads\WSI\test\W53-1-1-E.02.jpg');
wsi_SG=importdata('C:\Users\Andres\Downloads\SG\test\SG_W53-1-1-E.02.jpg');

% Scale factor
scale=4; 

% Scaled Images (WSI and Feature Annotation)
scaled_wsi = imresize(wsi,1/scale);
scaled_wsi_SG = imresize(wsi_SG,1/scale);

% Convert RGB image to indexed image

CTr=double(scaled_wsi_SG(:,:,1)==5 & scaled_wsi_SG(:,:,2)==208);
NEr=double(scaled_wsi_SG(:,:,1)==5 & scaled_wsi_SG(:,:,2)==5);
HBr=double(scaled_wsi_SG(:,:,1)==255 & scaled_wsi_SG(:,:,2)==102);
PCran=double(scaled_wsi_SG(:,:,1)==6 & scaled_wsi_SG(:,:,2)==208);
PCrnn=double(scaled_wsi_SG(:,:,1)==6 & scaled_wsi_SG(:,:,2)==4);
MVr=double(scaled_wsi_SG(:,:,1)==255 & scaled_wsi_SG(:,:,2)==51);

LEr = double(scaled_wsi_SG(:,:,1)==33 & scaled_wsi_SG(:,:,2)==143);
ITr=double(scaled_wsi_SG(:,:,1)==210 & scaled_wsi_SG(:,:,2)==5);

PCr = PCran | PCrnn;
CTr = LEr | ITr |CTr | MVr | HBr | PCr ;
mask = LEr | ITr | CTr | NEr | HBr | PCran | PCrnn | MVr;

SE=strel('disk',5);
CTr=imdilate(CTr,SE);

SE=strel('disk',5);
NEr=imdilate(NEr,SE);


% mask = CTr | NEr | HBr | PCr | MVr;

% close all;

% Suggestion: Shows in subplot scaled_wsi_SG and mask in order to see what
% is the right segmented area.

%%

[mask_pix_ct,mini_ct,~]=grountruthmap(CTr,scale);
[mask_pix_ne,mini_ne,~]=grountruthmap(NEr,scale);
% [mask_pix_hb,mini_hb,~]=grountruthmap(HBr,scale);
% [mask_pix_pc,mini_pc,~]=grountruthmap(PCr,scale);
% [mask_pix_mv,mini_mv,~]=grountruthmap(MVr,scale);


% Ground Truth Mask
% Each value correspond to label
grtruth_mask=2*mask_pix_ct+1*mask_pix_ne;

bw_mask=double(grtruth_mask>=1);

[~,mini,coord]=grountruthmap(bw_mask,scale);

%% Classification

classif_list=[];
predicted_list=[];

win_size=224/scale;

for i=1:size(coord,1)
    
    tissue=[];
    patch=[];
    
    patch=imcrop(wsi,[coord(i,2)*scale coord(i,1)*scale win_size*scale-1 win_size*scale-1]);
    
    tissue=double(patch(:,:,:))./255;
    
    label = classify(model,tissue);

    classif_list(i,1)=label;
    
end

classif_list=double(classif_list);

disp('Classification ended');


%% 
index=0;

classif_mask=zeros(size(grtruth_mask));
                
for i=1:size(coord,1)
    
            index=index+1;

            % Extracts patch coordinates

            % Build a mask with squared patches that will be extracted
            classif_mask(coord(i,1):coord(i,1)+win_size-1,coord(i,2):coord(i,2)+win_size-1)=classif_list(i,1);

end

disp('termina nuevo mapa');


%% 

% pp=grtruth_mask;

green = [0 1 0];
black = [0 0 0];
cyan = [0 1 1];
blue = [0 0 1];
% red = [1 0 0];


fuse_grtruth_mask=labeloverlay(scaled_wsi,double(grtruth_mask),'Colormap',[black;green],'Transparency',0.4);
fuse_classif_mask=labeloverlay(scaled_wsi,double(classif_mask),'Colormap',[black;green],'Transparency',0.4);

figure(1), 
subplot(1,3,1), imshow(scaled_wsi,[]); title('H&E');
subplot(1,3,2), imshow(fuse_grtruth_mask,[]); title('Groundtruth');
subplot(1,3,3), imshow(fuse_classif_mask,[]); title('Classification')

a=0;

%%

J = imresize(classif_mask,[size(wsi,1) size(wsi,2)]);
kk=mat2gray(J);
pp=uint8(kk*255);
imwrite(pp,'new_W53-1-1-E.02.png')


%% Classification performance (Filtered)

% classif_mask_mini


% [acc,spec,sens,p,n,tp,tn,fp,fn] = metrics2(grtruth_mask_mini,classif_mask_mini);
% 
% [acc2,spec2,sens2,p2,n2,tp2,tn2,fp2,fn2] = metrics2(grtruth_mask_mini,classif_mask_filt_mini);
% 
% table_val(1,:)=table(acc,spec,sens,p,n,tp,tn,fp,fn);
% table_val(2,:)=table(acc2,spec2,sens2,p2,n2,tp2,tn2,fp2,fn2);
% 
% writetable(table_val,'results.xlsx');

%

% [m,n]=size(classif_mask_mini)

% grtruth_vect=grtruth_mask_mini(:);
% class_filt_vect=classif_mask_filt_mini(:);
% 
% 
% 
% 
% p = sum(grtruth_vect==2)
% n = sum(grtruth_vect==1)
% 
% tp = sum(grtruth_vect==2 & class_filt_vect==2)
% tn = sum(grtruth_vect==1 & class_filt_vect==1)
% fp = sum(grtruth_vect==1 & class_filt_vect==2)
% fn = sum(grtruth_vect==2 & class_filt_vect==1)
% 
% acc = (tp+tn)/(p+n)
% spec = tp/(tp+fn)
% sens = tn/(tn+fp)
% % for i=1:m
%     for i=1:n
%         kk(i,j) = mgrtruth_mask_mini(i,j) == 2 & classif_mask_filt_mini(i,j) ==2;
%     end
% end



            
%% transform pixel-wise mask to segementation mask

% se=strel('disk',2);
% close_map_mini=imclose(classif_mask_mini,se);
% open_map_mini=imopen(classif_mask_mini,se);
% 
% close_mask = pixel2mask(close_map_mini,grtruth_mask,win_size);
% open_mask = pixel2mask(open_map_mini,grtruth_mask,win_size);
% fil_mask = pixel2mask(classif_mask_filt,grtruth_mask,win_size);
% 
% figure(3),
% subplot(2,2,1), imshow(grtruth_mask,[]); title('GroundTruth');
% subplot(2,2,2), imshow(fil_mask,[]); title('Gaussian Filt');
% subplot(2,2,3), imshow(close_mask,[]); title('Closing Operation');
% subplot(2,2,4), imshow(open_mask,[]); title('Opening Operation');

%% transform pixel-wise mask to segementation mask

%%
% filtered_mask = pixel2mask(classif_mask_filt_mini,grtruth_mask,win_size);
% fuse_filtered_classif_mask=labeloverlay(scaled_wsi,double(filtered_mask),'Colormap',[blue;green],'Transparency',0.5);

%%

% fuse_open_mask=labeloverlay(scaled_wsi,double(open_mask),'Colormap',[blue;green],'Transparency',0.5);
% fuse_close_mask=labeloverlay(scaled_wsi,double(close_mask),'Colormap',[blue;green],'Transparency',0.5);
% figure(5), 
% subplot(2,2,1), imshow(scaled_wsi,[]); title('H&E');
% subplot(2,2,2), imshow(fuse_grtruth_mask,[]); title('GroundTruth');
% subplot(2,2,3), imshow(fuse_classif_mask,[]); title('Classification');
% subplot(2,2,4), imshow(fuse_filtered_classif_mask,[]); title('Filtered Mask');


%% Other Functions

% Grid

function im_out=graph_grid(im_in,patchsize,esc,color)


[m,n] = size(im_in);

% tam_vent=200;    % El tamaño real es 10 veces más (2000 pix)
winsize=patchsize/esc;

col=floor(n/winsize);
row=floor(m/winsize);

im_in=imcrop(im_in,[1 1 winsize*col-1 winsize*row-1]);
% figure(1);
im_out=imshow(im_in,[],'InitialMagnification','fit');

hold on;

    for j=1:col

        % Grafica lineas verticales
        y=[1 m];
        x=[j*winsize j*winsize];
        hold on; plot(x,y,color);

    end

    for i=1:row
        % Grafica lineas horizontales
        x=[1 n];
        y=[i*winsize i*winsize];
        hold on; plot(x,y,color);
    end

end


% Crop Images (Function)

function [maskcoord,mini,coord] = grountruthmap(mask,scale)

    win_size=224/scale;


    cols=floor(size(mask,1)/win_size);
    rows=floor(size(mask,2)/win_size);

    maskcoord=zeros(size(mask,1),size(mask,2));

    index=0; % BEgin index with 0
    coord=[];

    for i=1:cols

        for j=1:rows


            mask_crop=imcrop(mask,[(j-1)*win_size+1 (i-1)*win_size+1 win_size-1 win_size-1]);

            calc_area=sum(mask_crop(:));
            porcentaje=(calc_area/(win_size*win_size))*100;

            mini(i,j)=0;

            % Tissue Percentage
            % Tenia esto en 95
            if porcentaje>90
                index=index+1;

                % Extracts patch coordinates

                coord(index,:)=[(i-1)*win_size+1 (j-1)*win_size+1];

                % Build a mask with squared patches that will be extracted
                maskcoord(coord(index,1):coord(index,1)+win_size-1,coord(index,2):coord(index,2)+win_size-1)=1;
                mini(i,j)=1;
            end

        end
    end
    
end


function [mask_m]=pixel2mask(class_map,maskcoord,win_size)

[m,n]=size(maskcoord);
mask_m=zeros(m,n);

[cols,rows]=size(class_map);

    for i=1:cols

        for j=1:rows

            mask_m((i-1)*win_size+1:(i-1)*win_size+win_size,(j-1)*win_size+1:(j-1)*win_size+win_size)=class_map(i,j);
            a=0;
        end

    end

a=0;
end

function [acc,spec,sens,p,n,tp,tn,fp,fn]=metrics2(grtruth_mask_mini,classif_mask_mini)

grtruth_vect=grtruth_mask_mini(:);
class_filt_vect=classif_mask_mini(:);

p = sum(grtruth_vect==2);
n = sum(grtruth_vect==1);

tp = sum(grtruth_vect==2 & class_filt_vect==2);
tn = sum(grtruth_vect==1 & class_filt_vect==1);
fp = sum(grtruth_vect==1 & (class_filt_vect==2 | class_filt_vect==0));
fn = sum(grtruth_vect==2 & (class_filt_vect==1 | class_filt_vect==0));

acc = (tp+tn)/(tp+tn+fp+fn);
spec = tp/(tp+fn);
sens = tn/(tn+fp);

end



