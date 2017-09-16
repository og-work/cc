%mnist data visualization
clear
clc
%close all

figure;
suptitle('Cross-encoder results');

k = 1;
ROWS=12;
COLS=10;

indices = [1 110 210 310 410 510 610 710 810 910];%randsample(COLS, 10000);

k = 1;  
offset = 0;
k = 1;
position = 0;

if 1
for digit1 = 1:5
    k = 1;
    digit2 = digit1+1;
    str_digit = num2str(digit1 - 1);
    str = strcat(str_digit, '_');
    str = strcat(str, num2str(digit2 - 1));
    filename = strcat('_debug_wtinit_1_feat_fusion_clsfr784_5_mnist_', str);
    filename1 = strcat(filename, '_valid_input.mat');
    filename2 = strcat(filename, '_valid_decoded.mat');
    decoded_input = load(filename2);
    input = load(filename1);
    
    for p = 1:COLS
        subplot(ROWS, COLS, offset + position + p);
        imshow((reshape(input.debug_valid_input((k), :), 28, 28))', []);
        subplot(ROWS, COLS, offset + position + p + COLS);
        imshow((reshape(decoded_input.debug_valid_decoded((k), :), 28, 28))',[]);
        %subplot(ROWS, COLS, offset + position + p + 2*COLS);
        %imshow((reshape(output_3_4.debug_valid_output((k), :), 28, 28))',[]);
        k = k + 100;
    end
    offset = offset + 2*COLS;
end
end
 
%% Auto encoders 
figure;
suptitle('Autoencoder results');
k = 1;
ROWS=10;
COLS=5;

offset = 0;
k = 1;
for digit = 2:10
    k = 1;
    str_digit = num2str(digit - 1);
    str = strcat(str_digit, '_');
    str = strcat(str, str_digit);
    filename = strcat('_debug_wtinit_1_feat_fusion_clsfr784_5_mnist_', str);
    filename1 = strcat(filename, '_valid_input.mat');
    filename2 = strcat(filename, '_valid_decoded.mat');
    decoded_input = load(filename2);
    input = load(filename1);
    
    for p = 1:COLS
        subplot(ROWS, COLS, offset + position + p);
        imshow((reshape(input.debug_valid_input((k), :), 28, 28))', []);
        subplot(ROWS, COLS, offset + position + p + COLS);
        imshow((reshape(decoded_input.debug_valid_decoded((k), :), 28, 28))',[]);
        %subplot(ROWS, COLS, offset + position + p + 2*COLS);
        %imshow((reshape(output_3_4.debug_valid_output((k), :), 28, 28))',[]);
        k = k + 10;
    end
    offset = offset + 2*COLS;
end
