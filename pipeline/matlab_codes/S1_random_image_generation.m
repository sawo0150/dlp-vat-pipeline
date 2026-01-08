// S1_random_image_generation.m

clc; clear; close all;
%% step 1 Random image generation with speckles
%% 2024-09-02 Ko seunggyu
% offset xy coordinate 1600 1600 2600 320
% offset xy coordinate 1600 1600 1208 135
% point xy coordinate 60 60 312 43  - 16 by 16
% point xy coordinate 60 60 312 43  - 32 by 32
% labview 400 400 2572 344  - 16 by 16
% labview 800 800 2452 220  - 32 by 32

sizeim =128;                % image size
t = 10005;                  % number of generated images
nam = "128randup";             % name of image
namingfile = "128randupbw";      % folder nam


if ~exist(namingfile,'dir')
    mkdir(namingfile)
    addpath(namingfile)
end
for tt=1:t
    A = uint8(zeros(sizeim,sizeim));
    num=randi([10 40],1);
    num2=randi([15 25],1);
    num3=randi([0 1],1);
    for k=1:num
        ty = randi([0 2],1);
        %gray = randi([0 255],1); % grayscale 
        gray = randi([0 1],1)*255; % binary 
        if ty==0
            x1 = randi([1 sizeim],1);
            y1 = randi([1 sizeim],1);
    %         x2 = randi([1 sizeim],1);
    %         y2 = randi([1 sizeim],1);
            r = randi([1 50],1);
            for i = 1:sizeim
                for j = 1:sizeim
                    if (x1-i)^2+(y1-j)^2 <= r^2
                        A(i,j)=gray;
                    end
                end
            end    
        end
        if ty==1
            x1 = randi([1 sizeim],1);
            y1 = randi([1 sizeim],1);
            x2 = randi([1 sizeim],1);
            y2 = randi([1 sizeim],1);
            x3 = randi([1 sizeim],1);
            y3 = randi([1 sizeim],1);
            X1 = [x1, y1];
            X2 = [x2, y2];
            X3 = [x3, y3];
            cos1 = dot(X3-X1,X2-X1)/(norm(X3-X1)*norm(X2-X1));
            cos2 = dot(X1-X2,X3-X2)/(norm(X1-X2)*norm(X3-X2));
            for i = 1:sizeim
                for j = 1:sizeim
                    Y = [i,j];
                    if (dot((X2-X1),(Y-X1))/(norm(X2-X1)*norm(Y-X1))>=cos1) &&...
                            (dot((X3-X1),(Y-X1))/(norm(X3-X1)*norm(Y-X1))>=cos1) &&...
                            (dot((X1-X2),(Y-X2))/(norm(X1-X2)*norm(Y-X2))>=cos2) &&...
                            (dot((X3-X2),(Y-X2))/(norm(X3-X2)*norm(Y-X2))>=cos2)
                        A(i,j)=gray;
                    end
                end
            end    
        end
        if ty==2
            x1 = randi([1 sizeim],1);
            y1 = randi([1 sizeim],1);
            x2 = randi([1 sizeim],1);
            y2 = randi([1 sizeim],1);
            h = randi([1 50],1);
            X1 = [x1, y1];
            X2 = [x2, y2];
            for i = 1:sizeim
                for j = 1:sizeim
                    Y = [i,j];
                    if (dot((X2-X1),(Y-X1))<=norm(X2-X1)*norm(X2-X1)) &&...
                            (dot((X2-X1),(Y-X1))>=0) &&...
                            (norm(Y-X2-dot((X1-X2),(Y-X2))/(norm(X1-X2))^2*(X1-X2))<=h)
                        A(i,j)=gray;
                    end
                end
            end    
        end
    end
    if num3 ==0 
        for k=1:num2
            gray1 = randi([0 1],1)*255; % binary
            %gray1 = randi([0 255],1); % grayscale
            ty = randi([0 1],1);
            x1 = randi([1 sizeim-1],1);
            y1 = randi([1 sizeim-1],1);
            if ty == 0
                A(x1,y1)=gray1;
            else
                A(x1,y1)=gray1;
                A(x1,y1+1)=gray1;
                A(x1+1,y1)=gray1;
                A(x1+1,y1+1)=gray1;
            end
        end
    else
    end
    filepath =strcat(pwd,'\',namingfile);
    filename=sprintf('%s%04d.png',nam,tt);
    imwrite(A,fullfile(filepath,filename),'png')
end

imshow(A)