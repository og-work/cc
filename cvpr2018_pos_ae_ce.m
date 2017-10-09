
close all
clc
figure;
ROWS = 10;
COLS = 10;
k =1;

for n = 1:9
   filename = strcat(strcat('/home/omkar/Documents/cvpr_cc_', num2str(n)), '_to_all') ;
   data = load(filename);
   for p = 0:9
   digit = function_get_digit(data, n, p);
   subplot(ROWS, COLS, k);
   imshow(digit, []);
   k = k +1;
   end
end