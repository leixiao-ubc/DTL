function [PSNR, PSNR_mean] = computePSNR()

global Training Data

PSNR = zeros(Training.N,1);

parfor i=1:Training.N
     residual_x = customCrop(Data{i}.ESTimg, Training.wpad) - Data{i}.GTimg;
     img_size = numel(residual_x(:));
     mse = sumsqr(residual_x)./img_size; 
     PSNR(i) = 20*log10(Training.data_normalization_value) - 10*log10(mse);
end

PSNR_mean = mean(PSNR(:));

end