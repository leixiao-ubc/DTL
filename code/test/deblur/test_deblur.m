%------------------------------------------------
%------------------------------------------------
% test code for paper "discriminative transfer learning for general image restoration"
% contact: Lei Xiao (leixiao08@gmail.com)
% copyright reserved
%------------------------------------------------
%------------------------------------------------

clear all;  close all; clc;

fprintf('\n--------------------------------------');
fprintf('\n--------------TESTING--------------');
fprintf('\n--------------------------------------\n');

% load trained model (lazy way)
global Model result_dir
load('../../../data/training_results_collection/result_2002/workspace.mat');
result_dir = strcat(result_dir, '_deblur');
mkdir(result_dir); 
fprintf('the results are saved in the folder: /%s\n', result_dir);
fprintf('remember to disable PSNR computation for run-time experiment.\n');

%%
global Test
Test = struct;
%*************************************
%*************************************
% choose input parameters below:
I = im2double(imread('../../../data/example_test_dataset/test002_gt.png'));
Test.kdims = [25,25]; 
load('../../../data/dataset_kernel_GP/dataset_test_psf_25_100.mat'); %PSFs
K = PSFs(:,:,1); % getch the psf
sigma = 1; % noise level
% index of the corresponding lambda: set offset = 100, 50, 0, when sigma=1, 3, 5 respectively
offset = 100; % change with sigma
tmp = expident(Model.cof(end-offset)); % weight of fidelity term
lambda_array = tmp.value;
%----------------------------------------------
%uncommend to use customized lambda value
%lambda_array = 500; % used for user-specified lambda value
%-----------------------------------------------
%*************************************
%*************************************

reset(RandStream.getGlobalStream);
Test.save_intermediate = false;
Test.prepad = false;
Test.use_quantized_meas = true;
Test.data_normalization_value = 255; %by default
Test.use_gpu = true;
Test.use_lut = true; 
if(Test.use_lut)
    compile;
end
Test.iter = Model.numStages; % should experiment with different iter
N = 1;
psnr_input = zeros(length(sigma), N);
psnr_output = zeros(length(sigma), N);

[Model] = test_precompute_lut(Model.cof(:), Model, Test.use_gpu);
 
%
for idx_noise = 1:length(sigma)
    
    reset(RandStream.getGlobalStream);

    Test.sigma = sigma(idx_noise);
    Test.offset = offset(idx_noise);
    lambda = lambda_array(idx_noise);
    
    % convolve with kernel and add noise
    ks = floor((size(K, 1) - 1)/2);
    yorig = I;
    y = conv2(yorig, K, 'valid');
    y = y + Test.sigma./255*randn(size(y));
    y = double(uint8(y .* 255))./255;
        
   DataOne.Meas = y;
   DataOne.GTpsf = K;
   DataOne.GTimg = customCrop(I, ks);

    fprintf('test noise level %f, lambda=%f\n', Test.sigma, lambda);
    
    idx_img = 1; 
    [Test] = test_loadData(Test, Model, DataOne);
    Test.crop_width = Test.wpad + floor((Test.kdims(1)-1)/2); % cropped each boundary for PSNR comparison
    [Model] = test_precompute_filters(Test, Model, Model.cof(:));
          
    Test.fn_img_meas_copy = sprintf('%s/test%03d_meas_noise%.4f.png', result_dir, idx_img, Test.sigma);
    imwrite(customCrop(Test.Meas./Test.data_normalization_value, Test.wpad), Test.fn_img_meas_copy);
    psnr_input(idx_noise, idx_img) = test_computePSNR(Test.Meas , Test.GT, Test.crop_width, Test.data_normalization_value);

    [Test] = test_computeLatentEstimation(Test, Model, lambda);      
          
    Test.fn_img_out = sprintf('%s/test%03d_result_noise%.4f_lambda%f.png', result_dir, idx_img, Test.sigma, lambda);
    imwrite(customCrop(Test.ESTimg./Test.data_normalization_value, Test.wpad), Test.fn_img_out);          
    psnr_output(idx_noise, idx_img) = test_computePSNR(Test.ESTimg, Test.GT, Test.crop_width, Test.data_normalization_value);
    fprintf('\tprocess image %02d, psnr (%f, %f)\n', idx_img, psnr_input(idx_noise, idx_img), psnr_output(idx_noise, idx_img));

end

psnr_mean = mean(psnr_output, 2);
fprintf('mean psnr is %.3f dB.\n', psnr_mean);


