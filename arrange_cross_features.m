%Code to extract and arrange AEC and CAE features propoerly

close all
clear all
clc

data = [1,1,1,2,2,2,3,3,3,4,4,4,12,12,12,13,13,13,14,14,14,...
        21,21,21,23,23,23,24,24,24,31,31,31,32,32,32,34,34,34, ...
        41,41,41,42,42,42,43,43,43];
n_classes = 4;
dim_feature = 3;
all_classfet = [];

for cl = 1:n_classes
   st_aec = (cl - 1) * dim_feature + 1;
   end_aec = st_aec + dim_feature - 1;
   aef = data(:, st_aec:end_aec);

   offset_cec = dim_feature * n_classes + ...
       (cl - 1)* dim_feature * (n_classes - 1) + 1;
   st_cec = offset_cec;
   end_cec = st_cec + dim_feature * (n_classes - 1) - 1;
   cef = data(:, st_cec:end_cec);
   classfet = [aef, cef];
   all_classfet = [all_classfet, classfet];
end
