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
load('../../../data/training_results_collection/result_2015/workspace.mat');
result_dir = strcat(result_dir, '_joint_inpaint_denoise');
mkdir(result_dir); 
fprintf('the results are saved in the folder: /%s\n', result_dir);
fprintf('remember to disable PSNR computation for run-time experiment.\n');

%%
global Test
Test = struct;
%*************************************
%*************************************
% choose input parameters below:
Test.path_img = './images';
Test.fn_img = 'boy';
Test.subsample_ratio = 0.39; % sample rate for measured pixels
sigma = [15]; % noise level for measured pixels
lambda_array = [6]; % user-specified lambda value (depends on both subsample_ratio and sigma)
%*************************************
%*************************************

Test.data_normalization_value = 255; %by default
Test.use_quantized_meas = false; % false following Chen's TRD paper
Test.kdims = [1,1];
Test.use_gpu = true;
Test.use_lut = true; 
if(Test.use_lut)
    compile;
end
Test.iter = Model.numStages; % should experiment with different iter
Test.crop_width = 10; % cropped each boundary for PSNR comparison
N = 1;
psnr_input = zeros(length(sigma), N);
psnr_output = zeros(length(sigma), N);

[Model] = test_precompute_lut(Model.cof(:), Model, Test.use_gpu);
 
for idx_noise = 1:length(sigma)
    
    reset(RandStream.getGlobalStream);
    
    Test.sigma = sigma(idx_noise);

    Test.fn_img_gt = sprintf('%s/%s.png', Test.path_img, Test.fn_img);
    Test.fn_img_meas_copy = sprintf('%s/%s_noise%d.png', result_dir, Test.fn_img, Test.sigma);
     
    [Test] = test_loadData(Test, Model);
    [Model] = test_precompute_filters(Test, Model, Model.cof(:));
    imwrite(Test.Meas./Test.data_normalization_value, Test.fn_img_meas_copy);
    psnr_input(idx_noise) = test_computePSNR(Test.Meas, Test.GT, Test.crop_width, Test.data_normalization_value);
    
    for idx_lambda = 1:length(lambda_array)        
        lambda = lambda_array(idx_lambda);
        fprintf('test noise level %f, lambda=%f\n', Test.sigma, lambda);

         [Test, psnr_temp] = test_computeLatentEstimation(Test, Model, lambda);     
         
         psnr_output(idx_noise, idx_lambda) = test_computePSNR(Test.ESTimg, Test.GT, Test.crop_width, Test.data_normalization_value);
         Test.fn_img_out = sprintf('%s/%s_out_lambda%f_psnr%.4f.png', result_dir, Test.fn_img, lambda, psnr_output(idx_noise, idx_lambda));
         imwrite(Test.ESTimg./Test.data_normalization_value, Test.fn_img_out);   
         
         fprintf('\nprocess image %s, psnr (%f, %f), lambda %.3f\n', Test.fn_img, psnr_input(idx_noise), psnr_output(idx_noise), lambda);
    end
end


psnr_mean = mean(psnr_output, 2);
fprintf('mean psnr is %.3f dB.\n', psnr_mean);


