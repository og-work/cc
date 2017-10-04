% Generate images for CVPR 2018
%clear 
close all

aec11_val_ip = load('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_1_1_valid_input');
aec10_val_dec = load('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_1_0_valid_decoded');
aec12_val_dec = load('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_1_2_valid_decoded');

% k = 5;
% subplot(1,3,1)
% img1_0 = imshow((reshape(aec11_val_ip.debug_valid_input(k, :), 28, 28))', []);
% subplot(1,3,2)
% img1_1 = imshow((reshape(aec10_val_dec.debug_valid_decoded(k, :), 28, 28))', []);
% subplot(1,3,3)
% img1_2 = imshow((reshape(aec12_val_dec.debug_valid_decoded(k, :), 28, 28))', []);

offset_y = 162;
offset_x = 120;
w = 54;
h = 54;
gap_y = 38;
gap_x = 10;

% figure;
% im11 = imread('CC_1_1_epoch_990.png');
% imshow('CC_1_1_epoch_990.png')
% cord_x = 9; cord_y = 1;
% patchx = offset_x + (cord_x - 1) * (w + gap_x);
% patchy = offset_y + (cord_y - 1) * (h + gap_y);
% startx = patchx;
% endx = patchx + w;
% starty = patchy;
% endy = patchy + h;
% patch_11 = im11(startx:endx, starty:endy, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% im20 = imread('CC_2_0_epoch_990.png');
% imshow(im20);
% digit_20 = function_crop_img(381, 920, w, h, im20);
% figure;imshow(digit_20);
% 
im21 = imread('CC_2_1_epoch_990.png');
imshow(im21)
digit_21 = function_crop_img(382, 636, w, h, im21);
figure;imshow(digit_21);

%im22 = imread('CC_2_2_epoch_990.png');
%imshow(im22);
%digit_22 = function_crop_img(186, 542, w, h, im22);
%figure;imshow(digit_22);

% im23 = imread('CC_2_3_epoch_990.png');
% imshow(im23);
% digit_23 = function_crop_img(381, 447, w, h, im23);
% figure;imshow(digit_23);

% im24 = imread('CC_2_4_epoch_990.png');
% imshow(im24);
% digit_24 = function_crop_img(382, 443, w, h, im24);
% figure;imshow(digit_24);

% im25 = imread('CC_2_5_epoch_990.png');
% imshow(im25);
% digit_25 = function_crop_img(381, 636, w, h, im25);
% figure;imshow(digit_25);

% im26 = imread('CC_2_6_epoch_990.png');
% imshow(im26);
% digit_26 = function_crop_img(381, 920, w, h, im26);
% figure;imshow(digit_26);

% im27 = imread('CC_2_7_epoch_990.png');
% imshow(im27);
% digit_27 = function_crop_img(381, 447, w, h, im27);
% figure;imshow(digit_27);

% im28 = imread('CC_2_8_epoch_990.png');
% imshow(im28);
% digit_28 = function_crop_img(381, 1014, w, h, im28);
% figure;imshow(digit_28);

% im29 = imread('CC_2_9_epoch_990.png');
% imshow(im29);
% digit_29 = function_crop_img(381, 731, w, h, im29);
% figure;imshow(digit_29);

save('/home/omkar/Documents/cvpr_cc_2_to_all')



