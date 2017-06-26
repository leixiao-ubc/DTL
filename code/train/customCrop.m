function [Y] = customCrop(X, wpad)

% INPUT:
% X: 3d matrix
% wpad: the number of pixels that need to be cropped out at each boundary
% side (only the first two dimensions in X)

% OUTPUT:
% Y: 3d matrix

Y = X(wpad+1:end-wpad, wpad+1:end-wpad, :);




end