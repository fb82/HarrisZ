function hz_demo
% HarrisZ and HarrisZ+ demo

% tested on Ubuntu 20.04

% you must have vlfeat installed for visualization, path must be set here 
vlfeat_path='/home/user/MATLAB Add-Ons/vlfeat-0.9.21'; 
run([vlfeat_path '/toolbox/vl_setup']);

%%% original HarrisZ
% read image
im=imread('graf5.png');
kpt_im_old=hz(im);
% show keypoints
figure;
imshow(im);
hold on;
vl_plotframe(kpt_im_old);
plot(kpt_im_old(1,:),kpt_im_old(2,:),'+r');

pause(0.5);
%%% HarrisZ+
% read image
im=imread('graf5.png');
% max_kpt is used for the spatial uniform ranking
max_kpts=8000;
% or
% max_kpts=2048;
kpt_im=hz_plus(im,max_kpts,0);
% all the keypoints are returned, the first max_kpt ones must be retained
kpt_im=kpt_im(:,1:min(size(kpt_im,2),max_kpts));
% show keypoints (origin of returned keypoints is in [0.5,0.5])
% so in the figure keypoints will be shifted)
figure;
imshow(im);
hold on;
vl_plotframe(kpt_im);
plot(kpt_im(1,:),kpt_im(2,:),'+r');

pause(0.5);
%%% HarrisZ+ for 
% another example with higher resolution image and more keypoints
% read image
im_big=imread('wooden_lady.jpg');
max_kpts=8000;
% or
% max_kpts=2048;
% set the second parameter to 1 in this case
% to avoid to get stuck and/or get out of memory
kpt_im_big=hz_plus(im_big,max_kpts,1);
% all the keypoints are returned, the first max_kpt ones must be retained
kpt_im_big=kpt_im_big(:,1:min(size(kpt_im_big,2),max_kpts));
% show keypoints
figure;
imshow(im_big);
hold on;
vl_plotframe(kpt_im_big);
plot(kpt_im_big(1,:),kpt_im_big(2,:),'+r');