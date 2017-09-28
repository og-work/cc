%mnist data visualization
clear
clc
%close all

indices = [1 110 210 310 410 510 610 710 810 910];%randsample(COLS, 10000);

if 0
    figure;
    suptitle('Cross-encoder results part1');
    ROWS=10;
    COLS=11;
    offset = 0;
    k = 1;
    for digit1 = 1:5
        k = 1;
        digit2 = digit1+1;
        str_digit = num2str(digit1 - 1);
        str = strcat(str_digit, '_');
        str = strcat(str, num2str(digit2 - 1));
        filename = strcat('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_', str);
        filename1 = strcat(filename, '_valid_input.mat');
        filename2 = strcat(filename, '_valid_decoded.mat');
        decoded_input = load(filename2);
        input = load(filename1);
        
        for p = 1:COLS-1
            subplot(ROWS, COLS, offset + p);
            imshow((reshape(input.debug_valid_input((k), :), 28, 28))', []);
            subplot(ROWS, COLS, offset + p + COLS);
            imshow((reshape(decoded_input.debug_valid_decoded((k), :), 28, 28))',[]);
            %subplot(ROWS, COLS, offset + position + p + 2*COLS);
            %imshow((reshape(output_3_4.debug_valid_output((k), :), 28, 28))',[]);
            k = k + 100;
        end
        str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
        str2 = strcat(str1, str);
        str3 = strcat(str2, '_mean_images');
        mean_images = load(str3);
        subplot(ROWS, COLS, offset + p + 1 + COLS);
        imshow((reshape(mean_images.debug_output_mean_image_train, 28, 28))',[]);
        
        valid_decoded_img = decoded_input.debug_valid_decoded((k), :);
        diff_img = valid_decoded_img - mean_images.debug_output_mean_image_train;
        subplot(ROWS, COLS, offset + p + 1 - 11 + COLS);
        imshow((reshape(diff_img, 28, 28))',[]);
        
        offset = offset + 2*COLS;
    end
end

if 0
    figure;
    suptitle('Cross-encoder results part2');
    ROWS=10;
    COLS=11;
    offset = 0;
    k = 1;
    for digit1 = 5:9
        k = 1;
        digit2 = digit1+1;
        str_digit = num2str(digit1 - 1);
        str = strcat(str_digit, '_');
        str = strcat(str, num2str(digit2 - 1));
        filename = strcat('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_', str);
        filename1 = strcat(filename, '_valid_input.mat');
        filename2 = strcat(filename, '_valid_decoded.mat');
        decoded_input = load(filename2);
        input = load(filename1);
        
        for p = 1:COLS-1
            subplot(ROWS, COLS, offset + p);
            imshow((reshape(input.debug_valid_input((k), :), 28, 28))', []);
            subplot(ROWS, COLS, offset + p + COLS);
            imshow((reshape(decoded_input.debug_valid_decoded((k), :), 28, 28))',[]);
            %subplot(ROWS, COLS, offset + position + p + 2*COLS);
            %imshow((reshape(output_3_4.debug_valid_output((k), :), 28, 28))',[]);
            k = k + 100;
        end
        
        str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
        str2 = strcat(str1, str);
        str3 = strcat(str2, '_mean_images');
        mean_images = load(str3);
        subplot(ROWS, COLS, offset + p + 1 + COLS);
        imshow((reshape(mean_images.debug_output_mean_image_train, 28, 28))',[]);
        
        valid_decoded_img = decoded_input.debug_valid_decoded((k), :);
        diff_img = valid_decoded_img - mean_images.debug_output_mean_image_train;
        subplot(ROWS, COLS, offset + p + 1 - 11 + COLS);
        imshow((reshape(diff_img, 28, 28))',[]);
        
        offset = offset + 2*COLS;
    end
end


if 1
    figure;
    suptitle('Cross-encoder results training as testing, random input ');
    ROWS=10;
    COLS=12;
    offset = 0;
    k = 1;
    for digit1 = 1:2
        k = 1;
        digit2 = digit1+1;
        str_digit = num2str(digit1 - 1);
        str = strcat(str_digit, '_');
        str = strcat(str, num2str(digit2 - 1));
        filename = strcat(strcat('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_', str), '_train_output_train_input_');
        filename1 = strcat(filename, 'input_perm.mat');
        filename2 = strcat(filename, 'train_decoded.mat');
        decoded_input = load(filename2);
        input = load(filename1);
        
        for p = 1:COLS-2
            subplot(ROWS, COLS, offset + p);
            imshow((reshape(input.debug_train_input_perm((k), :), 28, 28))', []);
            subplot(ROWS, COLS, offset + p + COLS);
            imshow((reshape(decoded_input.debug_train_decoded((k), :), 28, 28))',[]);
            %subplot(ROWS, COLS, offset + position + p + 2*COLS);
            %imshow((reshape(output_3_4.debug_valid_output((k), :), 28, 28))',[]);
            k = k + 100;
        end
        %        Mean image experiment >>>>>
                str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
                str2 = strcat(str1, str);
                str3 = strcat(str2, '_mean_images');
                mean_images = load(str3);
                subplot(ROWS, COLS, offset + p + 1 + COLS);
                imshow((reshape(mean_images.debug_output_mean_image_train, 28, 28))',[]);
        
                valid_decoded_img = decoded_input.debug_train_decoded((k-100), :);
                diff_img = valid_decoded_img - mean_images.debug_output_mean_image_train;
                subplot(ROWS, COLS, offset + p + 1 - 12 + COLS);
                imshow((reshape(diff_img, 28, 28))',[]);
        
        %Random input experiment
        random_sample_ind = 4;
        str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
        str2 = strcat(str1, str);
        str3 = strcat(str2, '_random_input');
        random_input = load(str3);
        subplot(ROWS, COLS, offset + p + 2 - 12 + COLS);
        imshow((reshape(random_input.debug_random_input(random_sample_ind, :), 28, 28))',[]);
        
        str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
        str2 = strcat(str1, str);
        str3 = strcat(str2, '_random_decoded');
        random_decoded = load(str3);
        subplot(ROWS, COLS, offset + p + 2  + COLS);
        imshow(reshape(random_decoded.debug_random_decoded(random_sample_ind, :), 28, 28)',[]);
        
        
        offset = offset + 2*COLS;
    end
end

%% Auto encoders

if 1
    figure;
    suptitle('Auto-encoder results training as validation, random inputs');
    ROWS=10;
    COLS=12;
    offset = 0;
    k = 1;
    for digit1 = 1:5
        k = 1;
        digit2 = digit1;
        str_digit = num2str(digit1 - 1);
        str = strcat(str_digit, '_');
        str = strcat(str, num2str(digit2 - 1));
        %filename = strcat(strcat('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_', str), '_train_output_train_input_');
        filename = strcat('_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_', str);
        %filename1 = strcat(filename, 'input_perm.mat');
        filename1 = strcat(filename, '_valid_input.mat');
        %filename2 = strcat(filename, 'train_decoded.mat');
        filename2 = strcat(filename, '_valid_decoded.mat');
        decoded_input = load(filename2);
        input = load(filename1);
        
        for p = 1:COLS-2
            subplot(ROWS, COLS, offset + p);
            imshow((reshape(input.debug_valid_input((k), :), 28, 28))', []);
            subplot(ROWS, COLS, offset + p + COLS);
            imshow((reshape(decoded_input.debug_valid_decoded((k), :), 28, 28))',[]);
            %subplot(ROWS, COLS, offset + position + p + 2*COLS);
            %imshow((reshape(output_3_4.debug_valid_output((k), :), 28, 28))',[]);
            k = k + 10;
        end
        %        Mean image experiment >>>>>
                str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
                str2 = strcat(str1, str);
                str3 = strcat(str2, '_mean_images');
                mean_images = load(str3);
                subplot(ROWS, COLS, offset + p + 1 + COLS);
                imshow((reshape(mean_images.debug_output_mean_image_train, 28, 28))',[]);
        
                valid_decoded_img = decoded_input.debug_valid_decoded((k-1), :);
                diff_img = valid_decoded_img - mean_images.debug_output_mean_image_train;
                subplot(ROWS, COLS, offset + p + 1 - 12 + COLS);
                imshow((reshape(diff_img, 28, 28))',[]);
        
        %Random input experiment
        random_sample_ind = 4;
        str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
        str2 = strcat(str1, str);
        str3 = strcat(str2, '_random_input');
        random_input = load(str3);
        subplot(ROWS, COLS, offset + p + 2 - 12 + COLS);
        imshow((reshape(random_input.debug_random_input(random_sample_ind, :), 28, 28))',[]);
        
        str1 = '_debug_wtinit_1_feat_fusion_clsfr784_25_mnist_';
        str2 = strcat(str1, str);
        str3 = strcat(str2, '_random_decoded');
        random_decoded = load(str3);
        subplot(ROWS, COLS, offset + p + 2 + COLS);
        imshow(reshape(random_decoded.debug_random_decoded(random_sample_ind, :), 28, 28)',[]);
        
        
        offset = offset + 2*COLS;
    end
end