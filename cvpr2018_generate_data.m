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


%load('/home/omkar/Documents/cvpr_cc_9_to_all');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% im10 = imread('CC_1_0_epoch_990.png');
% imshow(im10);
% digit_10 = function_crop_img(381, 542, w, h, im10);
% figure;imshow(digit_10);

% im11 = imread('CC_1_1_epoch_990.png');
% imshow(im11)
% digit_11 = function_crop_img(381, 352, w, h, im11);
% figure;imshow(digit_11);

% im12 = imread('CC_1_2_epoch_990.png');
% imshow(im12);
% digit_12 = function_crop_img(837, 258, w, h, im12);
% figure;imshow(digit_12);

% im13 = imread('CC_1_3_epoch_990.png');
% imshow(im13);
% digit_13 = function_crop_img(381, 1014, w, h, im13);
% figure;imshow(digit_13);

% im14 = imread('CC_1_4_epoch_990.png');
% imshow(im14);
% digit_14 = function_crop_img(381, 163, w, h, im14);
% figure;imshow(digit_14);

% im15 = imread('CC_1_5_epoch_990.png');
% imshow(im15);
% digit_15 = function_crop_img(381, 447, w, h, im15);
% figure;imshow(digit_15);

% im16 = imread('CC_1_6_epoch_990.png');
% imshow(im16);
% digit_16 = function_crop_img(381, 163, w, h, im16);
% figure;imshow(digit_16);

% im17 = imread('CC_1_7_epoch_990.png');
% imshow(im17);
% digit_17 = function_crop_img(837, 258, w, h, im17);
% figure;imshow(digit_17);

% im18 = imread('CC_1_8_epoch_990.png');
% imshow(im18);
% digit_18 = function_crop_img(381, 163, w, h, im18);
% figure;imshow(digit_18)

% im19 = imread('CC_1_9_epoch_990.png');
% imshow(im19);
% digit_19 = function_crop_img(381, 1014, w, h, im19);
% figure;imshow(digit_19);

save('/home/omkar/Documents/cvpr_cc_1_to_all')



