% Pick random patches

%  Esta es la siguiente versión (Branch2)

clc;
clear;
close all;

for ind=1:6
    switch ind

        case 1 % Leading Edge
            region = 'LE';
            disp(region);

        case 2
            region = 'IT';
            disp(region);
        
        case 3
            region = 'CT';
            disp(region);
            
        case 4
            region = 'HB';
            disp(region);

        case 5
            region = 'PC';
            disp(region);

        case 6
            region = 'MV';
            disp(region);
            
            
    end
    
    numpatches=copy_patches(region);
    disp(numpatches);
    
end

disp("fin del proceso");

    
function [numpatches] = copy_patches(region)
    
path_dir=['C:\Users\Andres\Desktop\val7\',region,'\'];
read_folder=dir(strcat(path_dir,'*.jpg'));

% Tomo sólo el 30% de los parches de cada carpeta
numpatches=round(size(read_folder,1)*0.10);

for i=1:numpatches
    
    rng('shuffle')
    randnum = randi([1 round(size(read_folder,1))],1);
    file_name=read_folder(randnum).name;
    
    SourceFile=['C:\Users\Andres\Desktop\val7\',region,'\',file_name];
    DestinyFile='C:\Users\Andres\Desktop\train8\CR2\';    
    
    copyfile(SourceFile, DestinyFile, 'f')

end

disp("fin del proceso")

end
