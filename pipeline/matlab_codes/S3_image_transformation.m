// S3_image_transformation.m

clc; clear; close all;
%% step 3 Postprocessing of captured image
%% step 4 Apply grayscale random threshold to multiple(single) images
%% 2024-09-02 Ko seunggyu

%% select mode
% mode 1 : Final postprocessing of camera image
% mode 2 : single image comparision
% mode 3 : rotation test
% mode 4 : padding code
% mode 5 : genereate threshold image
% mode 6 : single grayscale %%%%% important parameter

mode = 1;

    A = [cosd(-1.1) -sind(-1.1) 0; sind(-1.1) cosd(-1.1) 0; 0 0 1];
    tform = affinetform2d(A)
    AA = [1/9.98*8 0 0; 0 1/9.98*8 0; 0 0 1];
    tform1 = affinetform2d(AA)
    


if mode == 1

    namingfile = "bw_128011";      % folder name

    if ~exist(namingfile,'dir')
        mkdir(namingfile)
        addpath(namingfile)
    end
    filepath1 =strcat(pwd,'\',namingfile);
    % Iaa = rgb2gray(II);
    % surf(II)
    % shading interp
    % colorbar
    

    %
    for i=1:10000
        II = imread(sprintf('%d.bmp',i));
        II = II.';
        J = imwarp(II,tform);
        JJ = imwarp(J,tform1);
        %JJ = imtranslate(JJ,[-0.6, 0.3]);
        filepath =strcat(pwd,'\',namingfile);
        filename=sprintf('%d.png',i);

        %Jnew = padarray(JJ,[128 128],0,'both'); %% condition for 512 512
        %Jnew = padarray(JJ,[256 256],0,'both'); %% condition for 1024 1024
        %Jnew = padarray(JJ,[256 256],0,'both');  %% 1280 1280
        Jnew = padarray(JJ,[256 256],0,'both');  %% 1600 1600
        %Jcrop = imcrop(Jnew,[202 217 511 511]);   % condition for 512 512
        %Jcrop = imcrop(Jnew,[211 238 1023 1023]);   % condition for 1024 1024
        Jcrop = imcrop(Jnew,[267 286 1279 1279]);  %% condition for 1280 1280
        %Jcrop = imcrop(Jnew,[350 363 1023 1023]);  %% condition for 960 960

        % Jnew = padarray(JJ,[512 512],0,'both');  %% previous
        % Jcrop = imcrop(Jnew,[120 108 2047 2047]);  %% previous      
        %Jcrop = imcrop(Jnew,[118 95 255 255]); %% previous
        %Jcrop = imcrop(Jnew,[198 215 511 511]);  %% previous
        %2452 220 labview fromer condition
        %Jcrop = imcrop(Jnew,[220 222 511 511]);  %% previous
        % Jcrop = imcrop(Jnew,[97 110 255 255]);

        %193 labview 7  glass
        imwrite(Jcrop,fullfile(filepath,filename),'png')
    end
    
    
elseif mode == 2

    K = imread('1.png');
    I = imread('1.bmp');


    I = I.';
    J = imwarp(I,tform);

    % figure(1)
    % imagesc(J)
    JJ = imwarp(J,tform1);

    % figure(2)
    % imagesc(JJ)
    %Jnew = padarray(JJ,[128 128],0,'both');  %% 512 512
    %Jnew = padarray(JJ,[256 256],0,'both');  %% 1024 1024
    %Jnew = padarray(JJ,[256 256],0,'both');  %% 1280 1280
    %Jnew = padarray(JJ,[256 256],0,'both');  %% 1600 1600
    Jnew = padarray(JJ,[256 256],0,'both');  %% 980 960

    % Jcrop = imcrop(Jnew,[202 217 511 511]);  %% condition for 512 512
    %Jcrop = imcrop(Jnew,[211 238 1023 1023]);  %% condition for 512 512
    Jcrop = imcrop(Jnew,[187 186 1279 1279]);  %% condition for 1280 1280
    %Jcrop = imcrop(Jnew,[350 363 1279 1279]);  %% condition for 960 960

    
    %Jcrop = imcrop(Jnew,[120 108 2047 2047]);
    %Jcrop = imcrop(Jnew,[97 110 255 255]);
    %%%%%%%% Jcrop = imcrop(Jnew,[200 95 255 255]); %% condition for 400 400 2572
    %344 labview - glass_0
    %Jcrop = imcrop(Jnew,[118 95 255 255]);  %% condition for 400 400 2572
    %344 labview - glass_7
    %Jcrop = imcrop(Jnew,[202 217 511 511]);  %% condition for 800 800 2452
    %220 labview no glass
    %Jcrop = imcrop(Jnew,[215 193 511 511]);   % condition for 800 800 2452
    %193 labview 7  glass
    %imagesc(Jcrop)

    % figure(3)
    % imagesc(K)
    
    figure(4)
    % imshowpair(Jcrop,K,"diff")
    imshowpair(Jcrop,K)
    % % imwrite(Jcrop,"changed.png","png");
    
elseif mode ==3
    %% test code for rotation
    II = imread('4WALL.png');
    A = [cosd(1.1) -sind(1.1) 0; sind(1.1) cosd(1.1) 0; 0 0 1];
    tform = affinetform2d(A)

    AA = [cosd(-1.1) -sind(-1.1) 0; sind(-1.1) cosd(-1.1) 0; 0 0 1];
    tform1 = affinetform2d(AA)

    T = [1 0 4; 0 1 4; 0 0 1];
    tform2 = affinetform2d(AA)

    J = imwarp(II,tform);
    figure(5)
    imagesc(J)

    JJ = imwarp(J,tform1);
    JJJ = imcrop(JJ,[43 83 4199 2159]);
    KK=abs(II)-abs(JJJ);
    k = mean(mean(KK));
    % imshow(JJ)

    figure(1)
    imagesc(II)

    figure(2)
    imagesc(JJJ)

    figure(3)
    % imshowpair(JJJ,II,"diff")
    imshowpair(JJJ,II,"diff")

    figure(4)
    P = reshape(KK,2160*4200,1);
    Q = histogram(P,'Normalization','probability');

    
    
    %% sample code
    % I = imread('pixel.png');
    % % I = imread('png240.png');
    % Iimage = imcrop(I,[90 400 10 10]);
    % % imshow (Iimage);
    % Ia = rgb2gray(Iimage);
    % imagesc(Ia);
    % % surf(Ia);
    % % shading interp
    % colorbar
    % colormap(jet)
elseif mode ==4 
    %% padding code
    for ii=1:10000
        I = imread(sprintf('%d.png',ii));
        num = 1280;
    
        A = zeros(num,num);
    
        for i = 1:num;
            for j = 1:num;
                A(i,j)=I(ceil(i/8),ceil(j/8));
            end
        end
    
        % AA = padarray(A,[128 128],0,'both');
        imwrite(uint8(A),sprintf('3_%d.png',ii),"png");
    end
elseif mode ==5

    image_size =1280;
    namingfile = "train1";      % folder name
    if ~exist(namingfile,'dir')
        mkdir(namingfile)
        addpath(namingfile)
    end
    filepath1 =strcat(pwd,'\',namingfile);
    
    for ii=1:8000
        I = imread(sprintf('%d.png',ii));
        turn = randi([0 3],1);
        if turn ~=0
            maxi = max(max(I));
            %gray = randi([0 maxi],1);
            if maxi>100
                gray = randi([0 100],1);
            else
                gray = randi([0 maxi],1);
            end

            for i=1:image_size
                for j=1:image_size
    
                    if I(i,j)>=gray
                        I(i,j)=gray;
                    else
                        I(i,j)=0;
                    end
                end
            end
            filename=sprintf('%d.png',ii);
            imwrite(I,fullfile(filepath1,filename),'png')
        else
            gray = randi([0 100],1);
            for i=1:image_size
                for j=1:image_size
    
                    if I(i,j)>=gray
                        I(i,j)=gray;
                    else
                        I(i,j)=0;
                    end
                end
            end
            filename=sprintf('%d.png',ii);
            imwrite(I,fullfile(filepath1,filename),'png')
        end
    end

elseif mode ==6

    image_size =1280;
    namingfile = "bw_1280_complex";      % folder name
    if ~exist(namingfile,'dir')
        mkdir(namingfile)
        addpath(namingfile)
    end
    filepath1 =strcat(pwd,'\',namingfile);
    
    for ii=1:15
        I = imread(sprintf('traisn_%d.png',ii));
            gray = 220;
            for i=1:image_size
                for j=1:image_size
    
                    if I(i,j)>=gray
                        I(i,j)=100;
                    else
                        I(i,j)=0;
                    end
                end
            end
            filename=sprintf('m_%d.png',ii+48);
            imwrite(I,fullfile(filepath1,filename),'png')
    end

end

