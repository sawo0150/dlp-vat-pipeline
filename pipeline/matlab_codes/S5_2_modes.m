// S5_2_modes.m

clc; clear; close all;

%% step 7 Downsizing and upscaleing the image
%% 2024-09-02 Ko seunggyu

%% select mode
% mode 1 : image into 2^n size
% mode 2 : upscaling and padding of 2^n size image
% mode 3 : Threshold setting
% mode 4 : manipulate the trained image
% mode 5 : test for the trained image
% mode 6 : grayscale accuracy
% mode 7 : grayscale accuracy for test image
% mode 8 : original image accuracy
% mode 9 : change sample into window size of the image
% mode 10 : change default image into window size of the image
% mode 11 : grayscale up down scaling
% mode 12 : downsizeing with grayscale
% mode 13 : downsizing and make it only image without padding
% mode 14 : crop image

mode =2;

if mode ==1
%% different size
    image_size = 1024;
    % number = 3;
    num = 0;

    for number=15
        for k = 16:18
            % image_size = 2^k;
            I = imread(sprintf('%d.bmp',k));
            I = rgb2gray(I);
            J = imresize(I, 1/8,"nearest");

            for i=1:image_size/8
                for j=1:image_size/8
                    if J(i,j)>=128
                        J(i,j)=num;
                    else
                        J(i,j)=255-num;
                    end
                end
            end
            % figure(1)
            % imshow(J)
            M=J;
            imwrite(M,sprintf('%d.png',k));
        end
    end
elseif mode ==2
    %% padding
    image_size = 1024;
    for number =9:32
        I = uint8(double(imread(sprintf('02_%d.png',number)))*255);
        for i = 1:image_size
            for j = 1:image_size
                A(i,j)=I(ceil(i/8),ceil(j/8));
            end
        end
        AA = padarray(A,[image_size/8 image_size/8],0,'both');
        imwrite(uint8(AA),sprintf('4_%d.png',number));
    end
    % 
    % for number =16:18
    %     I = imread(sprintf('%d.bmp',number));
    %     I = im2gray(I);
    %         for i=1:image_size
    %             for j=1:image_size
    %                 if I(i,j)>=128
    %                     I(i,j)=0;
    %                 else
    %                     I(i,j)=255-0;
    %                 end
    %             end
    %         end
    %     AA = padarray(I,[image_size/8 image_size/8],0,'both');
    %     imwrite(uint8(AA),sprintf('train_%d.png',number));
    % end
    % 

    
elseif mode ==3
    %% Threshold setting
    for number = 1:100
        namingfile = sprintf('6',number);  % folder name
        image_size =1280
    
        if ~exist(namingfile,'dir')
            mkdir(namingfile)
            addpath(namingfile)
        end
        filepath1 =strcat(pwd,'\',namingfile);
        for gray = 30:30
            % I = imread(sprintf('grid%d.png',number));
            % I = im2gray(imread(sprintf('%d.png',gray));
            I = im2gray(imread(sprintf('07_%d_fake_B.png',number)));
            for i=1:image_size
                for j=1:image_size
                    if I(i,j)>=gray
                        I(i,j)=255;
                    else
                        I(i,j)=0;
                    end
                end
            end
            filename=sprintf('1_128_%d.png',number);
            imwrite(I,fullfile(filepath1,filename),'png')
        end
    end
        
elseif mode ==4
    %% nearest pooling and upscale
    
    image_size=512;
    for gray = [80 90 100 110 120 130 140 150 160 170 180 190 200 210]
        for number =1
            I = imread(sprintf('2input%d_256_%d_fake_B.png',number,gray));
            J = imresize(I, 1/8,"nearest");
            for i = 1:image_size
                for j = 1:image_size
                    A(i,j)=J(ceil(i/8),ceil(j/8));
                end
            end
            imwrite(uint8(A),sprintf('3input%d_32_%d.png',number,gray));
        end
    end
 elseif mode ==5   
    %% TEST
    clc; clear all; close all;
    image_size = 512;
    thresh =84; 
    I = imread(sprintf('90.png'));
    I = I(:,:,1);
    J=I;
    J=rot90(J,3);
    figure(1)
    imagesc(I)
    colormap(turbo)
    colorbar
    clim([0 255])
    for i=1:image_size
        for j=1:image_size
            if I(i,j)>=thresh
                I(i,j)=255;
            else
                I(i,j)=0;
            end
        end
    end
    imwrite(I,'1a.png','png');
    figure(2)
    imshow(I)

 elseif mode ==6

 image_size = 512;

 Max_index = 1:20;
 Max_output = 1:20;
    for ii= 1:20
        Acc = zeros(254,1);
        II = imread(sprintf('AA%d_32_real_A.png',ii));
        for thresh=1:254
            I = imread(sprintf('%d.png',ii+1));
            for i=image_size/4+1:image_size/4*3
                for j=image_size/4+1:image_size/4*3
                    if (double(I(i,j))-thresh-0.5)*(double(II(i,j))-thresh) <0
                        Acc(thresh) = Acc(thresh) + 1;
                    end
                end
            end
            thresh
            Acc(thresh)
        end
        Acc = Acc/256/256;
        [Max_index(ii),Max_output(ii)]=min(Acc);
    end
    Max_index = Max_index*256*256;
    % x=1:254;
    % plot(x,Acc)

 elseif mode ==7
 image_name = 1; 
 image_size = 512;
        Acc = zeros(26,1);
        II = imread(sprintf('AA%d_32_real_A.png',image_name));
        %II = imread(sprintf('B%d_32.png',image_name));
        for thresh=85:110
            I = imread(sprintf('%d.png',thresh-85+1+26*(image_name-1)));
            for i=image_size/4+1:image_size/4*3
                for j=image_size/4+1:image_size/4*3
                    if (double(I(i,j))-thresh-0.5)*(double(II(i,j))-thresh) <0
                        Acc(thresh-84) = Acc(thresh-84) + 1;
                    end
                end
            end
            thresh
            Acc(thresh-84)
        end
        Acc = Acc;
    x=85:110;
    plot(x,Acc)
    fontsize(gcf,scale=1.8)
    xlabel("trained threshold")
    ylabel("pixel with eror")

elseif mode ==8

 Acc = zeros(254,20);
 for image_name =1:20
    image_size = 512;
        II = imread(sprintf('AA%d_32_real_A.png',image_name));
        for thresh=1:254
            %I = imread(sprintf('%d.png',image_name));
            I = imread(sprintf('%d.png',image_name));
            for i=image_size/4+1:image_size/4*3
                for j=image_size/4+1:image_size/4*3
                    if (double(I(i,j))-thresh-0.5)*(double(II(i,j))-thresh) <0
                        Acc(thresh,image_name) = Acc(thresh,image_name) + 1;
                    end
                end
            end
            thresh
            Acc(thresh,image_name)
        end
        Acc = Acc;
 end
 
    % x=1:254;
    % plot(x,Acc)
    % fontsize(gcf,scale=1.8)
    % xlabel("threshold")
    % ylabel("pixel with eror")

elseif mode ==9
    sizeim =64;                  % image size
    t = 520;                   % number of generated images
    xcor = 880;                  % x coordinate loaction
    ycor = 640;                  % y coordinate location
    nam = "64rand";              % name of former image
    namingfile2 = "glass_new";       % former file name
    nam2 = "window_32";              % current image name
    namingfile = "glasss";  % folder name
    
    if ~exist(namingfile,'dir')
        mkdir(namingfile)
        addpath(namingfile)
    end
    
    filepath1 =strcat(pwd,'\',namingfile);
    filepath2 =strcat(pwd,'\',namingfile2,'\');
    k=0;
    for iname=1:20
        for gray = 85:110
            name =sprintf('%s2input%d_32_%d_fake_B_fake_B.png',filepath2,iname,gray);
            I = importdata(name);
            J = imresize(I, 1/8,"nearest");
            A = zeros(1080,1920);
            for i=1:sizeim
                for j=1:sizeim
                      A(i+ycor-sizeim/4,j+xcor-sizeim/4) = J(i,j);
                end
            end
            k=k+1;
            filename=sprintf('%s%04d.png',nam2,k);
            imwrite(uint8(A),fullfile(filepath1,filename),'png')
        end
    end

elseif mode ==10
    sizeim =64;                  % image size
    t = 520;                   % number of generated images
    xcor = 880;                  % x coordinate loaction
    ycor = 640;                  % y coordinate location
    nam = "64rand";              % name of former image
    namingfile2 = "subpixel"       % former file name
    nam2 = "window_32"              % current image name
    namingfile = "32subpixel2mask";  % folder name
    
    if ~exist(namingfile,'dir')
        mkdir(namingfile)
        addpath(namingfile)
    end
    
    filepath1 =strcat(pwd,'\',namingfile);
    filepath2 =strcat(pwd,'\',namingfile2,'\');
    k=0;
    for iname=1:20
        for gray = 85:110
            name =sprintf('%s3input%d_256_%d.png',filepath2,iname,gray);
            I = importdata(name);
            J = imresize(I, 1/8,"nearest");
            A = zeros(1080,1920);
            for i=1:sizeim
                for j=1:sizeim
                      A(i+ycor-sizeim/4,j+xcor-sizeim/4) = J(i,j);
                end
            end
            k=k+1;
            filename=sprintf('%s%04d.png',nam2,k);
            imwrite(uint8(A),fullfile(filepath1,filename),'png')
        end
    end
elseif mode == 11
    image_size=1280;
    for gray = 1
        for number =1:92
            I = im2gray(imread(sprintf('07_%d_fake_B.png',number)));
            %I = (imread(sprintf('2_%d_fake_B.png',number)));
            J = imresize(I, 1/8,"nearest");
            for i = 1:image_size/8
                for j = 1:image_size/8
                    if J(i,j)>=30
                        J(i,j)=255;
                    else
                        J(i,j)=0;
                    end
                end
            end
            for i = 1:image_size
                for j = 1:image_size
                    A(i,j)=J(ceil(i/8),ceil(j/8));
                end
            end
            K = imcrop(J,[17 17 127 127]);

            imwrite(uint8(A),sprintf('01_%d.png',number));
        end
    end

    namingfile = sprintf('graysacle_input_253');  % folder name
elseif mode ==12
    namingfile = sprintf('bw_input_1280');  % folder name
    if ~exist(namingfile,'dir')
        mkdir(namingfile)
        addpath(namingfile)
    end
    filepath1 =strcat(pwd,'\',namingfile);
    
    image_size=1280;
        for gray = 1:100
                I = imread(sprintf('01_%d_fake_B.png',gray));
                J = imresize(I, 1/8,"nearest");
                for i = 1:image_size
                    for j = 1:image_size
                        if J(ceil(i/8),ceil(j/8))>29
                            A(i,j)=255;
                        else
                            A(i,j)=0;
                        end
                    end
                end
                filename=sprintf('02_%d.png',gray);
                imwrite(A,fullfile(filepath1,filename),'png')
        end
elseif mode ==13
    image_size=1280;
    for gray = 1
        for number =1:88
            I = imread(sprintf('1_%d_fake_B.png',number));
            J = imresize(I, 1/8,"nearest");
            for i = 1:image_size/10
                for j = 1:image_size/10
                    if J(16+i,16+j)>253
                        A(i,j)=255;
                    else
                        A(i,j)=0;
                    end
                end
            end
            imwrite(uint8(A),sprintf('3_253_%d.png',number));
        end
    end
end
