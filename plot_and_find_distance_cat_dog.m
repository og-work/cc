clc

BASE_PATH = '/home/SharedData/omkar/data/'
data_exp4 = 'apy_50_400_cc1_data_part4_';
data_exp5 = 'apy_50_400_cc1_data_part5_';

train_classes = [7, 8, 10, 12, 13, 17, 21, 22, 23, 24];

margin = 0.0;
p = 1;

if 1
for j = 1:length(train_classes)
    for k = 1:length(train_classes)
        if (j ~= k)
            str0 = strcat(BASE_PATH, data_exp4);
            str1 = strcat(str0, num2str(train_classes(j)));
            str2 = strcat(str1, '_');
            str3 = strcat(str2, num2str(train_classes(k)));
            str4 = strcat(str3, '.mat');            
            cc1 = load(str4);
            
            str0 = strcat(BASE_PATH, data_exp5);
            str1 = strcat(str0, num2str(train_classes(j)));
            str2 = strcat(str1, '_');
            str3 = strcat(str2, num2str(train_classes(k)));
            str4 = strcat(str3, '.mat');
            io = load(str4);
            
            ip = io.cc1_input_valid';%(sample, :);
            op = io.cc1_output_valid';%(sample, :);
            decoded_ip = cc1.decoded_data_valid_cc1';%(sample, :);
            
            s_dim = 1;
            e_dim = size(ip, 1);
                        
            distance_ip_and_decoded_ip = sqrt(sum((ip(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :)).*(ip(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :))));
            distance_op_and_decoded_ip = sqrt(sum((op(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :)).*(op(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :))));
            req_distances = zeros(size(distance_op_and_decoded_ip));
            req_distances(distance_op_and_decoded_ip + margin < distance_ip_and_decoded_ip) = 1;
            results(p, 3) = sum(req_distances)/length(req_distances);
            results(p, 1) = train_classes(j);
            results(p, 2) = train_classes(k);
                      
            sprintf('Classes [%d %d]', train_classes(j), train_classes(k))
            sprintf('Max distances: ip-decoded-ip %f, op-decoded-ip %f, margin %f, Acc:%f', ...
                max(distance_ip_and_decoded_ip), max(distance_op_and_decoded_ip), margin, results(p, 3))
             p = p + 1;
        end
    end
end
figure;
stem(results(:, 3)'); title('validation results');
result = results(:, 3)';
result(result < 0.7) = 0;
result(result > 0.7) = 1;
sum(result)
end

if 0
for j = 1:length(train_classes)
    for k = 1:length(train_classes)
        if (j ~= k)
            str0 = strcat(BASE_PATH, data_exp);
            str1 = strcat(str0, num2str(train_classes(j)));
            str2 = strcat(str1, '_');
            str3 = strcat(str2, num2str(train_classes(k)));
            str4 = strcat(str3, '.mat');            
            cc1 = load(str4);
            
            str0 = strcat(BASE_PATH, data_exp);
            str1 = strcat(str0, num2str(train_classes(j)));
            str2 = strcat(str1, '_');
            str3 = strcat(str2, num2str(train_classes(k)));
            str4 = strcat(str3, '.mat');
            io = load(str4);
            
            ip = io.cc1_input_train_ori(sample, :)';
            op = io.cc1_output_train_ori(sample, :)';
            decoded_ip = cc1.decoded_data_train_cc1(sample, :)';
                        
            distance_ip_and_decoded_ip = sqrt(sum((ip(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :)).*(ip(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :))));
            distance_op_and_decoded_ip = sqrt(sum((op(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :)).*(op(s_dim:e_dim, :) - decoded_ip(s_dim:e_dim, :))));
            req_distances = zeros(size(distance_op_and_decoded_ip));
            req_distances(distance_op_and_decoded_ip + margin < distance_ip_and_decoded_ip) = 1;
            results1(p, 3) = sum(req_distances)/length(req_distances);
            results1(p, 1) = train_classes(j);
            results1(p, 2) = train_classes(k);
                      
            sprintf('Classes [%d %d]', train_classes(j), train_classes(k))
            sprintf('Max distances: ip-decoded-ip %f, op-decoded-ip %f, margin %f, Acc:%f', ...
                max(distance_ip_and_decoded_ip), max(distance_op_and_decoded_ip), margin, results(p, 3))
             p = p + 1;
        end
    end
end
end

figure;
if 0
        
    stem(ip(s_dim:e_dim), 'r', 'filled');hold on
    stem(op(s_dim:e_dim), 'g', 'filled');hold on
    stem(decoded_ip(s_dim:e_dim), 'b', 'filled');
else
    plot(ip(s_dim:e_dim), 'r');hold on
    plot(op(s_dim:e_dim), 'g');hold on
    plot(decoded_ip(s_dim:e_dim), 'b');
end
title('classes: car/cat :red: input(e.g. cat), green: output(e.g. dog), blue: decoded input (e.g. decoded cat)')




