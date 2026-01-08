// S4_image_padding.m

clc; clear ; close all;

%% step 5 Padding of origianl mask
%% 2024-09-02 Ko seunggyu
%% select mode
% mode 1 : Padding of muliple images
% mode 2 : padding and upscalling of single image
% mode 3 : padding
% mode 4 : individual image padding

mode = 1;

namingfile = "128grid_final";       % folder name
image_size = 1024;          % 128-->1280으로 변환 

if ~exist(namingfile,'dir')
    mkdir(namingfile)
    addpath(namingfile)
end
filepath1 =strcat(pwd,'\',namingfile);

%% padding code

if mode == 1
    for ii=1:18
        I = uint8(255*double(imread(sprintf('rec5_%d.png',ii))));
        %I = imread(sprintf('128randup%04d.png',ii));
        A = zeros(image_size,image_size);
    
        for i = 1:image_size
            for j = 1:image_size
                A(i,j)=I(ceil(i/8),ceil(j/8));
            end
        end
        filepath =strcat(pwd,'\',namingfile);
        filename=sprintf('rec1_%d.png',ii);
        imwrite(A,fullfile(filepath,filename),'png')
   
        AA = padarray(A,[image_size/8 image_size/8],0,'both');
        % imwrite(uint8(AA),"pool.png","png");
        imwrite(uint8(AA),fullfile(filepath,filename),'png')
    end

%% single image
elseif mode == 2
    I = imread('0_16.png');   
        for i = 1:image_size
            for j = 1:image_size
                A(i,j)=I(ceil(i/8),ceil(j/8));
            end
        end
    imwrite(uint8(A),"01_16.png",'png');


elseif mode == 3
    I = imread('0_16.png');
    A = padarray(I,[image_size/8 image_size/8],0,'both');
    imwrite(uint8(A),"01_16.png",'png');

elseif mode == 4
        ii = 1:8;
        %I = imread(sprintf('%d.png',ii));
        I = imread(sprintf('rec5_%d.png',ii));
        A = zeros(image_size,image_size);
    
        for i = 1:image_size
            for j = 1:image_size
                A(i,j)=I(ceil(i/8),ceil(j/8));
            end
        end
        filepath =strcat(pwd,'\',namingfile);
        filename=sprintf('wangle%d.png',ii);
        imwrite(A,fullfile(filepath,filename),'png')
   
        AA = padarray(A,[image_size/8 image_size/8],0,'both');
        % imwrite(uint8(AA),"pool.png","png");
        imwrite(uint8(AA),fullfile(filepath,filename),'png')

end