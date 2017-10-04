function [cropped_image] = function_crop_img(x, y, w, h, img)

startx = x;
starty = y;
endx = startx + w;
endy = starty + h;

cropped_image = img(startx:endx, starty:endy, :);


