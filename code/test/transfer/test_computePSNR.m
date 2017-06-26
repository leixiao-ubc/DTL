function [psnr] = test_computePSNR(meas, ref, crop_width, max_value)

residual_x = customCrop(meas-ref, crop_width);
img_size = numel(residual_x(:));
mse = sumsqr(residual_x)./img_size; 
psnr = 20*log10(max_value) - 10*log10(mse);

