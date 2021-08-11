%{

Author: Andres Sandino
Created: 07/05/2021

Source: 
On Stain normalization in deep learning
http://www.andrewjanowczyk.com/on-stain-normalization-in-deep-learning/

%}


function []=FastStainFunction(path,destpath,ImRef)

% ImRef: Reference Image

readfolder=dir(strcat(path,'*.jpg'));


    for numsample = 1:size(readfolder,1)

        % Read Image filename
        filename = readfolder(numsample).name;     
        % Read Image
        ImSample = importdata([path,filename]);    
        % Normalized image (Function faststain)
        ImOut = faststain(ImSample,ImRef);
        % Save image
        imwrite(ImOut,[destpath,filename]);
        %disp(numsample)

    end
    

end

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

