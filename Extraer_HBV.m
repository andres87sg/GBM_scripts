%%%%%%%%%
% Fecha creación: 9/Nov/2019
% Fecha modificado: 11/Abr/2020
% Abrir archivos WSI y extraer parches cuadrados
% 
% Nota: la imagen original está en im_in=imread(file_name);
%%%%%%%%%%
%asg
clc;
clear;
close all;

%% Main 

main_path='/home/usuario/Documentos/GBM/IvyGap/'
path_wsi=[main_path,'/WSI/test/'];
path_segmentation=[main_path,'/SG/test/'];

read_folder=dir(strcat(path_wsi,'*.jpg'));

% mex ./stain_normalization_toolbox/colour_deconvolution.c;
% addpath('./stain_normalization_toolbox')  


for num_case=1:size(read_folder,1) % Testing
% for num_case=1:1 % Testing

    
    disp('***************');
    disp(['Caso numero ',num2str(num_case)])
    disp('***************');
    
    file_name=read_folder(num_case).name; 
    file_name=file_name(1:size(file_name,2)-4);
    info_patches=croppatches(file_name,path_wsi,path_segmentation);
    
    table_patches(num_case,:)=[file_name,info_patches(:,2)'];

    Name = table_patches(:,1);
%     LE = table_patches(:,2);
%     IT = table_patches(:,3);
      CT = table_patches(:,4);
%     NE = table_patches(:,5);
      HB = table_patches(:,6);
%     PC = table_patches(:,7);
%     MV = table_patches(:,8);
    
    num_paches_table=table(Name,CT,HB);
%     num_paches_table=table(Name,CT);
%     a=0;
%     num_paches_table=table(Name,LE,IT,CT,NE,HB,PC,MV);
  
   writetable(num_paches_table,'/home/usuario/Documentos/GBM/TablePatches3.xlsx','Sheet','test');

end

disp("The process has ended")

%% Function Crop patches

function [info_patches]=croppatches(subblock_id,path_dir_wsi,path_dir_segmentation)


% wsi: Whole Slide Image || wsi_SG: Whole Slide Image Segmentation

wsi=importdata([path_dir_wsi,subblock_id,'.jpg']); %WSI 
wsi_SG=importdata([path_dir_segmentation,'SG_',subblock_id,'.jpg']);

% Scale factor
scale=2; 

% Scaled Images (WSI and Feature Annotation)
% scaled_wsi = imresize(wsi,1/scale);
scaled_wsi_SG = imresize(wsi_SG,1/scale);

% for ind=3:3
% for ind=[3,5]
for ind=[3,5]
% for ind=3:3
    
    coord=[];
%     path_region=[];
    
    switch ind

        case 1 % Leading Edge

            region = 'LE';
            LEr = double(scaled_wsi_SG(:,:,1)==33 & scaled_wsi_SG(:,:,2)==143);
            
            stride=224;
            [~,coord] = crop_patches(LEr,scale,stride);
            

        case 2 % Infiltrating Tumor (IT)

            region = 'IT';
            ITr=double(scaled_wsi_SG(:,:,1)==210 & scaled_wsi_SG(:,:,2)==5);
            
            stride=224;
            [~,coord] = crop_patches(ITr,scale,stride);
            
        case 3 % Celular Tumor (CT)

            region = 'CT';            
            CTr=double(scaled_wsi_SG(:,:,1)==5 & scaled_wsi_SG(:,:,2)==208);
            
            stride = 224; %Test / Valid  
            %stride = 224*3; %Train  
            ws = 224;
            
            [~,coord] = crop_patches(CTr,scale,stride,ws,region);

        case 4 % Necrosis (NE)

            region = 'NE';
            NEr=double(scaled_wsi_SG(:,:,1)==5 & scaled_wsi_SG(:,:,2)==5);
            
            stride=224;
            [~,coord] = crop_patches(NEr,scale,stride);
            
        case 5 % Hyperplastic blood vessels (HBV)

            region = 'HB'; 
            HBr=double(scaled_wsi_SG(:,:,1)==255 & scaled_wsi_SG(:,:,2)==102);
                        
            %stride = round(224*3/4); %Train
            stride = 224; % Valid / TEst
            ws = 224;
            [~,coord] = crop_patches(HBr,scale,stride,ws,region);
            
        case 6 % Pseudopalisading cells

            region = 'PC';
            
            stride=224;
            
            % Pseudopalisading cells around necrosis (CTpan)
            PCran=double(scaled_wsi_SG(:,:,1)==6 & scaled_wsi_SG(:,:,2)==208);
            [~,coord_an] = crop_patches(PCran,scale,stride); 
%             [PCran_maskcoord,coord_an] = crop_patches(PCran,scale,stride); 
            
            % Pseudopalisading cells but no visible necrosis (CTpnn)
            PCrnn=double(scaled_wsi_SG(:,:,1)==6 & scaled_wsi_SG(:,:,2)==4);
            [~,coord_nn] = crop_patches(PCrnn,scale,stride);  
%             [PCrnn_maskcoord,coord_nn] = crop_patches(PCrnn,scale,stride);  
            
            %maskcoord = PCrnn_maskcoord | PCran_maskcoord;
            coord=[coord_an;coord_nn];

            
        case 7 % Microvascular

            region = 'MV';
            MVr=double(scaled_wsi_SG(:,:,1)==255 & scaled_wsi_SG(:,:,2)==51);
            
            stride=224;
            [~,coord] = crop_patches(MVr,scale,stride);
            
    end
    
    wsi_SG_HB=double(wsi_SG(:,:,1)==255 & wsi_SG(:,:,2)==102); 
    %%%% Saving Patches
    path_region = ['/home/usuario/Documentos/GBM/Samples/samples_224x224/test/',region,'/'];
    %path_region_SG = ['C:\Users\Andres\Desktop\PatchesGBM\train16\',region,'_SG\'];
        
    save_patches(wsi,coord,ws,scale,path_region,subblock_id,region) 
    %save_patches_SG(wsi_SG_HB,coord,ws,scale,path_region_SG,subblock_id,region) 
    
    info_patches{ind,2}=size(coord,1);
    info_patches{ind,1}=region;
    
    %(wsi,coord,ws,scale,path,sub_block,region)
    clear coord path_region
    
end

TableInfoPatches=table(info_patches(:,1),info_patches(:,2));


disp('------------------------');
disp('Number of extracted patches');
disp(TableInfoPatches)
disp(['from: ',subblock_id,' at ',num2str(scale)]);
disp('------------------------');

end

% FIN DEL CÓDIGO


%% Other Functions

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

%%

function [maskcoord,coord] = crop_patches(mask,scale,stride,samplesize,region)

% OUT: [maskcoord: BW image , coord: patches coordinates]

win_size=samplesize/scale;
stride_size=stride/scale;


rows=floor(size(mask,1)/win_size);
cols=floor(size(mask,2)/stride_size);

maskcoord=zeros(size(mask,1),size(mask,2));

index=0; % BEgin index with 0
coord=[];

    for i=1:rows

        for j=1:cols


            mask_crop=imcrop(mask,[(j-1)*stride_size+1 (i-1)*win_size+1 win_size-1 win_size-1]);
           
            calc_area=sum(mask_crop(:));
            porcentaje=(calc_area/(win_size*win_size))*100;

            % Tissue Percentage
            switch region 
                
                case 'CT'
                    porcentaje_region=90;
                case 'HB'
                    porcentaje_region=30;
            end
            

            % tenia esto en 95
            if porcentaje>porcentaje_region
%             if porcentaje>10
                index=index+1;

                % Extracts patch coordinates

                coord(index,:)=[(i-1)*win_size+1 (j-1)*stride_size+1];
                % Build a mask with squared patches that will be extracted
                maskcoord(coord(index,1):coord(index,1)+win_size-1,coord(index,2):coord(index,2)+win_size-1)=1;

            end

        end
    end
    
    % Garantiza que la mascara de segmentación tenga el mismo tamaño a la
    % entrada y la salida
    maskcoord=imcrop(maskcoord,[1 1 size(mask,2)-1 size(mask,1)-1]);

%     a=0;
    
    
end


%%

function [] = save_patches(wsi,coord,ws,scale,path,sub_block,region)

win_size = ws/scale;

% mex ./stain_normalization_toolbox/colour_deconvolution.c;
% addpath('./stain_normalization_toolbox')  
% ref=imread('./stain_normalization_toolbox/ref2.jpg');

%path_norm = [path(1:length(path)-1),'norm\'];

    for i=1:size(coord,1)

        % Recuerde que _CT/_Ot depende de la carpeta
        
        patch=imcrop(wsi,[coord(i,2)*scale coord(i,1)*scale ...
                            win_size*scale-1 win_size*scale-1]);
        
        %verbose = 0;

        %patch_norm = Macenko(patch, ref, 255, 0.15, 1, verbose);
        
        
        [len,wid,~]=size(patch);
        
        if (len==wid)
        
            %if len==224*2}
            if len==224
        
            %patch_resize = imresize(patch,1/2);
            patch_resize = patch;
            %patch_norm_resize = imresize(patch_norm,1/2);


            name=[sub_block,'_',num2str(i),'_',region,'.jpg']; %%%% CAMBIAR ESTO (OJO!!!)

            imwrite(patch_resize,strcat(path,name),'jpg');
            %imwrite(patch_norm_resize,strcat(path_norm,name),'jpg');    
            
            end
        end
        


    end

end

function [] = save_patches_SG(wsi,coord,ws,scale,path,sub_block,region)

win_size = ws/scale;

    for i=1:size(coord,1)

        % Recuerde que _CT/_Ot depende de la carpeta

        patch=imcrop(wsi,[coord(i,2)*scale coord(i,1)*scale win_size*scale-1 win_size*scale-1]);
        name=[sub_block,'_',num2str(i),'_',region,'_mask.jpg']; %%%% CAMBIAR ESTO (OJO!!!)

        imwrite(patch,strcat(path,name),'jpg');

    end

end
