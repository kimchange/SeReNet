% By kimchange 2021 0107
addpath('./utils')
rng(0); % 0 for train 1 for test
Nnum = 13;
MLPitch =   100*1e-6; %% pitch of the microlens pitch
% M = 72.5;
% M = 22.56;
M = 63;
pixelPitch = MLPitch/Nnum;

xy_pixel_pitch = pixelPitch / M * 1e6; % xy pixel size is 0.12210 um
z_pixel_pitch = 0.2; % z pixel size is 0.2 um



synthetic.Height = 1989; % volume height
synthetic.Width = 1989; % volume width

synthetic.Depth = 101; %  volume depth


% scanning index and no-scanning index2
index1=[3:13:1989];
index2=[7:13:1989];
% index2=[4:7:1989*7/13];
index3=[11:13:1989];
% index=sort([index1,index2,index3]);

GPU = 1;
gpuDevice(1);
psf_param.Experimental_psf = 0;
% psf_param.Experimental_psf = 1;
% tag = 'test';
% tag = 'train-20231015v2-Ipsf-M63-aber0.2-resize7-101layers';
% tag = 'train-20231204-Ipsf-M63-aber0-resize13-101layers';
tag = ''
% disp(['tag = ',tag])
% savefolder = ['/media/aa/hdd/kimchange/synthetic/'];
savefolder = ['../'];
load("../../../psf/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u.mat");


% output direction
mkdir([savefolder,tag,'/GT_synthetic'])
mkdir([savefolder,tag,'/LF_synthetic'])
mkdir([savefolder,tag,'/x1_synthetic'])
mkdir([savefolder,tag,'/x3_synthetic'])

tic


num_lines_list = [linspace(100, 80, 6),(1:9)*0];

num_bubbles_list = [(1:6)*0, linspace(200, 100, 3),(1:6)*0];

num_beads_list = [(1:9)*0, linspace(5000, 1000, 5), 0];

num_big_beads_list = [(1:14)*0, 50];

bubble_diameter_list = [20,35,50,20,35,50,20,35,50,20,35,50,20,35,50];

% blur_sigmas_list = [linspace(0.5, 1.5, 6), linspace(0.5, 1.5, 3), linspace(0.5, 1.5, 3), linspace(0.5, 2.5, 3)];
blur_sigmas_list = [linspace(0.5, 1.5, 6), linspace(0.5, 1.5, 3), linspace(0.5, 2.5, 6)];



for this = 1:15


volume = zeros(synthetic.Height,synthetic.Width, round(synthetic.Depth * z_pixel_pitch / xy_pixel_pitch));
% num_lines = (11-this)*10;
num_lines = num_lines_list(this);

if num_lines > 0
    x_index = max( round((synthetic.Height - 200)*rand(3,num_lines)),1) + 100;
    y_index = max( round((synthetic.Width - 200)*rand(3,num_lines)),1) + 100;
    z_index = max( round(round(synthetic.Depth * z_pixel_pitch / xy_pixel_pitch)*rand(3,num_lines)),1);
    linelen = sqrt((x_index(1,:) - x_index(3,:)).^2 + ( y_index(1,:) - y_index(3,:) ).^2 + ( z_index(1,:) - z_index(3,:) ).^2 );
    center_bias = linelen .* tan(pi/6) / 2 .* ( 2*rand(1,num_lines) - 1 );
    x_index(2,:) = min( max( ( x_index(1,:) + x_index(3,:) ) /2 + center_bias , 1) , synthetic.Height);
    y_index(2,:) = min( max( ( y_index(1,:) + y_index(3,:) ) /2 + center_bias , 1) , synthetic.Width);
    z_index(2,:) = min( max( ( z_index(1,:) + z_index(3,:) ) /2 + center_bias , 1) , round(synthetic.Depth * z_pixel_pitch / xy_pixel_pitch));

    x_index = reshape(x_index, 1,[]);
    y_index = reshape(y_index, 1,[]);
    z_index = reshape(z_index, 1,[]);
    % figure;hold on

    for line = 1:num_lines
        [X,Y,Z] = bezier3(x_index(line*3-2:line*3), y_index(line*3-2:line*3), z_index(line*3-2:line*3));
        ind = sub2ind([synthetic.Height,synthetic.Width,round(synthetic.Depth * z_pixel_pitch / xy_pixel_pitch)], round(X),round(Y),round(Z));

        mag = rand()*500;
        intensity = 20 + mag;
        volume(ind) = volume(ind) + intensity;
    end


    % radius_pixel_num = floor( 3 / 2);
    radius_pixel_num = ceil(this/2);

    se = strel('sphere',radius_pixel_num);
    volume = imdilate(volume, se);

end
volume = imresize3(volume, [synthetic.Height, synthetic.Width, synthetic.Depth]); % 

% bubble 
% num_bubbles = 10+this;
num_bubbles = num_bubbles_list(this);

if num_bubbles > 0
    bubble_diameter = bubble_diameter_list(this);

    if (this<11) & (num_bubbles > 0)
        for bbb = 1:20
            rxy = floor( (bubble_diameter + bbb/2 - 5) / 2) ;
            rz =  floor( (bubble_diameter + bbb/2 - 5) * xy_pixel_pitch/ z_pixel_pitch / 2);
            delta_z = floor( (bubble_diameter + bbb/2 - 5) * xy_pixel_pitch/ z_pixel_pitch / 2 /2*3);
            delta_x = floor( (bubble_diameter + bbb/2 - 5) / 2 * 3 *(3)^0.5 / 2 );

            x_index = round((synthetic.Height - 200)*rand(1) + 100 ).* [1,1,1];
            y_index = round((synthetic.Width - 300)*rand(1) + 100 ).* [1,1,1]; 
            y_index(3) = y_index(3) + delta_x;
            z_center = (synthetic.Depth+1)/2 + floor(rand(1)*5-2);
            z_index = [z_center-delta_z, z_center+delta_z, z_center];
            ind = sub2ind([synthetic.Height,synthetic.Width,synthetic.Depth], round(x_index),round(y_index),round(z_index));

            relativeinds = getbubbleinds(rxy,rz,synthetic.Height,synthetic.Width,synthetic.Depth);

            for indidx = 1:length(ind)
                inds = ind(indidx) + relativeinds;
                volume(inds) = volume(inds) + (  round(20 + rand()*200) + zeros(size(relativeinds))  );
            end
        end
    end

    x_index = max( round((synthetic.Height - 200)*rand(1,num_bubbles)),1) + 100;
    y_index = max( round((synthetic.Width - 200)*rand(1,num_bubbles)),1) + 100;
    % z_index = round(linspace(25,bubbles.Depth-25,num_bubbles) + rand(1,num_bubbles)*10-5 );
    z_index = round(linspace(20,synthetic.Depth-20,num_bubbles) + rand(1,num_bubbles)*8-4 );

    index_c = round(linspace(1,num_bubbles+1,10));
    for idx = 1:length(index_c)-1
        cstart = index_c(idx);
        cend = index_c(idx+1)-1;
        ind = sub2ind([synthetic.Height,synthetic.Width,synthetic.Depth], round(x_index(cstart:cend)),round(y_index(cstart:cend)),round(z_index(cstart:cend)));
        rxy = floor( (bubble_diameter + idx - 5) / 2) ;
        rz =  floor( (bubble_diameter + idx - 5) * xy_pixel_pitch/ z_pixel_pitch / 2);
        relativeinds = getbubbleinds(rxy,rz,synthetic.Height,synthetic.Width,synthetic.Depth);
        % inds = ind + relativeinds;
        % volume(inds) = volume(inds) + round(20 + rand(size(ind))*200) + zeros(size(relativeinds));
        for indidx = 1:length(ind)
            inds = ind(indidx) + relativeinds;
            volume(inds) = volume(inds) + (  round(20 + rand()*200) + zeros(size(relativeinds))  );
        end


    end

end

num_beads = num_beads_list(this);
if num_beads > 0
    x_index = max( round((synthetic.Height - 200)*rand(1,num_beads)),1) + 100;
    y_index = max( round((synthetic.Width - 200)*rand(1,num_beads)),1) + 100;
    % z_index = round(linspace(15,synthetic.Depth-15,num_beads) + rand(1,num_beads)*8-4 );
    z_index = max( round((synthetic.Depth - 20)*rand(1,num_beads)),1) + 10;

    index_c = round(linspace(1,num_beads+1,6));
    for beadr = 1:length(index_c)-1
        cstart = index_c(beadr);
        cend = index_c(beadr+1)-1;

        ind = sub2ind([synthetic.Height,synthetic.Width,synthetic.Depth], round(x_index(cstart:cend)),round(y_index(cstart:cend)),round(z_index(cstart:cend)));
        % rxy = floor(bead_diameter / xy_pixel_pitch / 2);
        % rz = floor(bead_diameter / z_pixel_pitch / 2);
        rxy = beadr; rz = ceil(beadr / 2);
        relativeinds = getbeadinds(rxy,rz,synthetic.Height,synthetic.Width,synthetic.Depth); %  relativeinds = relativeinds(2:end-1);
        % inds = ind + relativeinds;
        % volume(inds) = volume(inds) + round(40 + rand(size(ind))*200) + zeros(size(relativeinds));
        for indidx = 1:length(ind)
            inds = ind(indidx) + relativeinds;
            volume(inds) = volume(inds) + (  round(40 + rand()*200) + zeros(size(relativeinds))  );
        end

    end
end

% big beads
% num_big_beads = 8;
num_big_beads = num_big_beads_list(this);

if num_big_beads > 0
    x_index = max( round((synthetic.Height - 300)*rand(1,num_big_beads)),1) + 150;
    y_index = max( round((synthetic.Width - 300)*rand(1,num_big_beads)),1) + 150;
    % z_index = round(linspace(25,bubbles.Depth-25,num_big_beads) + rand(1,num_big_beads)*10-5 );
    % z_index = round(linspace(40,synthetic.Depth-40,num_big_beads) + rand(1,num_big_beads)*20-10 );
    z_index = max( round((synthetic.Depth - 60)*rand(1,num_big_beads)),1) + 30;

    index_c = round(linspace(1,num_big_beads+1,10));
    for beadr = 1:length(index_c)-1
        cstart = index_c(beadr);
        cend = index_c(beadr+1)-1;
        
        ind = sub2ind([synthetic.Height,synthetic.Width,synthetic.Depth], round(x_index(cstart:cend)),round(y_index(cstart:cend)),round(z_index(cstart:cend)));
        rxy = 8*beadr; rz = beadr;
        relativeinds = getbeadinds(rxy,rz,synthetic.Height,synthetic.Width,synthetic.Depth);
        % inds = ind + relativeinds;
        % volume(inds) = volume(inds) + round(10 + rand(size(ind))*100) + zeros(size(relativeinds));
        for indidx = 1:length(ind)
            inds = ind(indidx) + relativeinds;
            volume(inds) = volume(inds) + (  round(10 + rand()*100) + zeros(size(relativeinds))  );
        end
    end
end

% gaussian blur
sigma_this = blur_sigmas_list(this);
if sigma_this > 0
    GaussM = fspecial3('gaussian', ceil([sigma_this*6+1,sigma_this*6+1,7]), [sigma_this, sigma_this, sigma_this / 2]); % different level of blur kernel
% volume = volume + 20 + 3*randn(size(volume));
    volume = convn(volume, GaussM, 'same');
end

volume = uint16(volume); % better
% plot3(X,Y,Z,'r')
% mkdir('/media/liuyv/denseTubes/GT_synthetic')
% imwrite3d(volume,[savefolder,tag,'/GT_synthetic/',num2str(this,'%03d'),'synthetic_size_',num2str(size(volume),'%d_'),'linesNum_',num2str(num_lines, '%d_'),datestr(now,'yyyy_mmdd_HHMMSS'),'.tif'],16)
imwrite3d(volume,[savefolder,tag,'/GT_synthetic/group',num2str(this,'%03d'),'.tif'],16);
tt = toc;
disp([num2str(this,'%03d'),' volume done, took ',num2str(tt),' seconds']);


[img_r,img_c,allz] = size(volume);
[psf_r,~,~,Nnum,psf_z] = size(psf);




LF = zeros(img_r,img_c,Nnum,Nnum,'single');
% volume = gpuArray(single(volume));
volume_fft = gather(fftn(ifftshift(gpuArray(single(volume)) )));
for u=1:size(psf,3)
    for v = 1:size(psf,4)
        fsp = gpuArray.zeros(img_r,img_c,psf_z,'single');% flip z and padded version of psf
        fsp((img_r+1)/2-(psf_r-1)/2:(img_r+1)/2+(psf_r-1)/2,(img_c+1)/2-(psf_r-1)/2:(img_c+1)/2+(psf_r-1)/2,:)  = flip(gpuArray(single(squeeze(psf(:,:,u,v,:)))),3);
        fsp = gather(fftn(ifftshift(fsp)));
        sumupXG1 = gpuArray( single(sum(  (volume_fft).*fsp,3)) );
        LF(:,:,u,v)=gather(abs(fftshift(ifftn(sumupXG1)))./psf_z);
    end
end


LF = permute(LF,[1,2,4,3]);
LF = reshape(LF,[size(LF,1),size(LF,2),size(LF,3)*size(LF,4)]);
max(LF(:))
LF = gather(LF);
x1 = LF(index2, index2, :);
% x3 = LF(index, index, :);
x3 = imresize(LF, [length(index2)*3, length(index2)*3] );

% mkdir('./LF_synthetic')
% imwriteTFSK(single(LF),[savefolder,tag,'/LF_synthetic/group',num2str(this,'%03d'),'.tif']);
% imwriteTFSK(single(x1),[savefolder,tag,'/x1_synthetic/group',num2str(this,'%03d'),'_1x1.tif']);
% imwriteTFSK(single(x3),[savefolder,tag,'/x3_synthetic/group',num2str(this,'%03d'),'_3x3.tif']);

imwrite3d(uint16(LF),[savefolder,tag,'/LF_synthetic/group',num2str(this,'%03d'),'.tif'],16);
imwrite3d(uint16(x1),[savefolder,tag,'/x1_synthetic/group',num2str(this,'%03d'),'_1x1.tif'],16);
imwrite3d(uint16(x3),[savefolder,tag,'/x3_synthetic/group',num2str(this,'%03d'),'_3x3.tif'],16);
tt = toc;
disp([num2str(this,'%03d'),' imaging done, took ',num2str(tt),' seconds']);

end