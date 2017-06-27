%------------------------------------------------
%------------------------------------------------
% test code for paper "discriminative transfer learning for general image restoration"
% contact: Lei Xiao (leixiao08@gmail.com)
% copyright reserved
%------------------------------------------------
%------------------------------------------------

clear all;  close all; clc;

if ~isdeployed  
    addpath(genpath('BM3D'));
    addpath(genpath('BM3D_images'));
end


fprintf('\n--------------------------------------');
fprintf('\n--------------TESTING--------------');
fprintf('\n--------------------------------------\n');

% load trained model (lazy way)
global Model result_dir
load('../../../data/training_results_collection/result_2015/workspace.mat');
result_dir = strcat(result_dir, '_modular');
mkdir(result_dir); 
fprintf('the results are saved in the folder: /%s\n', result_dir);
fprintf('remember to disable PSNR computation for run-time experiment.\n');

%%
global Test
Test = struct;
%*************************************
%*************************************
% choose input parameters below:
Test.path_img = './BM3D_images';
Test.fn_img = 'man';
sigma = [25]; %noise level
lambda_array = [0.4]; % lambda value
Test.init_rho_bm3d = 1;
Test.rho_ratio_bm3d = 2;
Test.lambda_bm3d = 20;
%*************************************
%*************************************

Test.kdims = [1,1];
Test.save_intermediate = true;
Test.data_normalization_value = 255; %by default
Test.use_quantized_meas = false; 
Test.use_gpu = true;
Test.use_lut = true; 
if(Test.use_lut)
    compile;
end
Test.iter = Model.numStages; % should experiment with different iter
Test.crop_width = 10; % cropped each boundary for PSNR comparison
start_idx = 1;
N = 1-start_idx+1;
psnr_input = zeros(length(sigma), N);
psnr_output = zeros(length(sigma), N);

[Model] = test_precompute_lut(Model.cof(:), Model, Test.use_gpu);
[Model] = test_precompute_filters(Model, Model.cof(:));
    
for idx_noise = 1:length(sigma)
    
    reset(RandStream.getGlobalStream);
    
    Test.sigma = sigma(idx_noise);
           
    Test.fn_img_gt = sprintf('%s/%s.png', Test.path_img, Test.fn_img);
    Test.fn_img_meas_copy = sprintf('%s/%s_noise%d.png', result_dir, Test.fn_img, Test.sigma);
    [Test] = test_loadData(Test, Model);

    imwrite(Test.Meas./Test.data_normalization_value, Test.fn_img_meas_copy);
    psnr_input(idx_noise) = test_computePSNR(Test.Meas, Test.GT, Test.crop_width, Test.data_normalization_value);
    
    for idx_lambda = 1:length(lambda_array)        
        lambda = lambda_array(idx_lambda);
        fprintf('test noise level %f, lambda=%f\n', Test.sigma, lambda);
        
         [Test, psnr_temp] = test_computeLatentEstimation(Test, Model, lambda);      
         
         Test.fn_img_out = sprintf('%s/%s_out_lambda%f.png', result_dir, Test.fn_img, lambda);
         imwrite(Test.ESTimg./Test.data_normalization_value, Test.fn_img_out);    
         psnr_output(idx_noise, idx_lambda) = test_computePSNR(Test.ESTimg, Test.GT, Test.crop_width, Test.data_normalization_value);
         fprintf('\nprocess image %s, psnr (%f, %f), lambda %.3f\n', Test.fn_img, psnr_input(idx_noise), psnr_output(idx_noise), lambda);
    end
end


psnr_mean = mean(psnr_output, 2);
fprintf('mean psnr is %.3f dB.\n', psnr_mean);

