%visualising cross-encoded data
%close all
clear all
clc

%valid_labels = load('mnist_98_784_feat_fusion_clsfr_valid_labels');
%test_labels = load('mnist_39_784_feat_fusion_clsfr_test_labels');
%n_valid_samples = size(valid_labels.valid_labels,2);

valid_data = load('mnist_5_784_cnn_svm_784_dim_valid_data');
indices = [1 2000 4600 6386 7570 9560 11000 12716 15000 17000];

%Display cross-encoder outputs
if 1
    for cc = 0:0
        figure;
        k = 1;
        ROWS=10;
        COLS=10;
        endI = 0;
        offset = 0;
        isDisplayed = 0;
        k = 1;%train_data = load('mnist_39_784_cec_features_class_tr_5');
        
        filename = strcat('mnist_5_784_cec_features_class_vl_', num2str(cc));
        title_string = strcat('Cross coder from input class: ', num2str(cc));
        suptitle(title_string);
        
        valid_data_1 = load(filename);
        
        for digit1 = 1:8
            
            startI = endI + 1;
            endI = startI + 784 - 1;
            if isDisplayed == 0
                isDisplayed = 1;
                for p = 1:COLS
                    subplot(ROWS, COLS, p);
                    imshow((reshape(valid_data.valid_data(indices(p), :), 28, 28))', []);
                end
            end
            
            offset = offset + p;
            
            for p = 1:COLS
                subplot(ROWS, COLS, offset + p);
                imshow((reshape(valid_data_1.cross_feautures_val(indices(p), startI:endI), 28, 28))',[]);
                image_array(k, :) = valid_data_1.cross_feautures_val(indices(p), startI:endI);
                k = k +1;
            end
            
        end
    end
end
%Display Autoencoder outputs

if 1
    figure;
    k = 1;
    ROWS=11;
    COLS=10;
    endI = 0;
    offset = 0;
    isDisplayed = 0;
    k = 1;%train_data = load('mnist_39_784_cec_features_class_tr_5');
    
    filename = strcat('mnist_5_784_aec_features_all_classes_vl_', num2str(9));
    title_string = strcat('Auto encoder from input class: ', num2str(9));
    suptitle(title_string);
    
    valid_data_1 = load(filename);
    
    for digit1 = 1:10
        
        startI = endI + 1;
        endI = startI + 784 - 1;
        if isDisplayed == 0
            isDisplayed = 1;
            for p = 1:COLS
                subplot(ROWS, COLS, p);
                imshow((reshape(valid_data.valid_data(indices(p), :), 28, 28))', []);
            end
        end
        
        offset = offset + p;
        
        for p = 1:COLS
            subplot(ROWS, COLS, offset + p);
            imshow((reshape(valid_data_1.cross_feautures_val(indices(p), startI:endI), 28, 28))',[]);
            image_array(k, :) = valid_data_1.cross_feautures_val(indices(p), startI:endI);
            k = k +1;
        end
        
    end
end



