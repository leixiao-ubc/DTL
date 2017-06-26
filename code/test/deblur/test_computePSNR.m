function [psnr] = test_computePSNR(meas, ref, crop_width, max_value)


crop_meas = customCrop(meas, crop_width);
crop_ref = customCrop(ref, crop_width);

residual_x = crop_meas-crop_ref;
img_size = numel(residual_x(:)); 
mse = sumsqr(residual_x)./img_size; 


psnr = 20*log10(max_value) - 10*log10(mse);

