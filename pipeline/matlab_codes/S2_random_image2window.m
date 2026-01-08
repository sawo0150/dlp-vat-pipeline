// S2_random_image2window.m

clc; clear; close all;
%% step 2 Random image to window mask (1920*1080)
%% 2024-09-02 Ko seunggyu
sizeim =128;                  % image size
t = 300;                    % number of generated images
xcor = 852;                   % x coordinate loaction    % 850 880
ycor = 640;                   % y coordinate location    % 610 640
nam = "01_";            % name of former image
namingfile2 = "correct"     % former file name
nam2 = "window"               % current image name
namingfile = "mask_after"; % folder name


if ~exist(namingfile,'dir')
    mkdir(namingfile)
    addpath(namingfile)
end

filepath1 =strcat(pwd,'\',namingfile);
filepath2 =strcat(pwd,'\',namingfile2,'\');

for tt=10:t
    %name =sprintf('%s%s%04d.png',filepath2,nam,tt);
    name =sprintf('%s%s%d.png',filepath2,nam,tt);
    I = uint8(double(importdata(name))*255); 
    %I = importdata(name);
    A = uint8(zeros(1080,1920));
    for i=1:sizeim
        for j=1:sizeim
              A(i+ycor,j+xcor) = I(i,j);
        end
    end
    filename=sprintf('%s%04d.png',nam2,tt);
    imwrite(uint8(A),fullfile(filepath1,filename),'png')
    %imwrite(A,fullfile(filepath1,filename),'png')
end