function [y] = imfilter_warp(x, f)

y = imfilter(x, f, 'conv', 'circular');

end