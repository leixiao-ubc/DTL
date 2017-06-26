%------------------------------------------------
%------------------------------------------------
% test code for paper "discriminative transfer learning for general image restoration"
% contact: Lei Xiao (leixiao08@gmail.com)
% copyright reserved
%------------------------------------------------
%------------------------------------------------

clear all;  close all; clc;

if ~isdeployed  
   addpath(genpath('subaxis'));
end

fprintf('\n--------------------------------------');
fprintf('\n--------------TESTING--------------');
fprintf('\n--------------------------------------\n');

% load trained model (lazy way)
global Model result_dir
load('../../../data/training_results_collection/result_2001/workspace.mat');
mkdir(result_dir); 
fprintf('the results are saved in the folder: /%s\n', result_dir);
fprintf('remember to disable PSNR computation for run-time experiment.\n');

%%
global Test
Test = struct;
%*************************************
%*************************************
% choose input parameters below:
Test.path_img = '../../../data/example_test_dataset/';
N = 5; % number of images to test
start_idx = 1; % index of the first image to test
sigma = [15]; % noise level
% index of the corresponding lambda: set offset = 320, 240, 160, 80, 0 when sigma=5, 10, 15, 20, 25 respectively
offset = [160]; % change with sigma 
%*************************************
%*************************************


Test.data_normalization_value = 255; %by default
Test.use_quantized_meas = false; % following Chen's TRD paper for fair comparison
Test.save_intermediate = false;
Test.use_lut = true; 
Test.use_gpu = true; % only used to precompute the look-up-table.

if(Test.use_lut)
    compile;
end
Test.iter = Model.numStages; % should experiment with different iter
Test.crop_width = 10; % cropped each boundary for PSNR comparison

lambda_array = [0];
psnr_input = zeros(length(sigma), N);
psnr_output = zeros(length(sigma), length(lambda_array), N);

[Model] = test_precompute_lut(Model.cof(:), Model, Test.use_gpu);
[Model] = test_precompute_filters(Model, Model.cof(:));
 
for idx_noise = 1:length(sigma)
    for idx_lambda = 1:length(lambda_array)
        
    reset(RandStream.getGlobalStream);
    
    Test.sigma = sigma(idx_noise);
    Test.offset = offset(idx_noise);
    tmp = expident(Model.cof(end-Test.offset)); % weight of fidelity term
    lambda = tmp.value; 
    layerscof = reshape(Model.cof(1:Model.len_cof_shared), Model.len_layercof, Model.numDenoiseLayers); 
    
    %use customized lambda
    %lambda =  lambda_array(idx_lambda);
    
    fprintf('test noise level %f, lambda=%f\n', Test.sigma, lambda);
    
    for idx_img = start_idx:start_idx+N-1 % each image might have different resolution, need to redo precomputation  
          Test.fn_img_gt = sprintf('%s/test%03d_gt.png', Test.path_img, idx_img);
          
          [Test] = test_loadData(Test, Model);
          Test.fn_img_meas_copy = sprintf('%s/test%03d_meas_noise%.4f.png', result_dir, idx_img, Test.sigma);
          imwrite(Test.Meas./Test.data_normalization_value, Test.fn_img_meas_copy);          
          psnr_input(idx_noise, idx_img) = test_computePSNR(Test.Meas , Test.GT, Test.crop_width, Test.data_normalization_value);
          %Test.fn_img_meas_copy = sprintf('%s/test%03d_meas_noise%f_psnr%.4f.png', result_dir, idx_img, Test.sigma, psnr_input(idx_noise, idx_img));
          %imwrite(Test.Meas./Test.data_normalization_value, Test.fn_img_meas_copy);
          Test.fn_img_interm = sprintf('%s/test%03d_result_noise%.4f_lambda%f_iter', result_dir, idx_img, Test.sigma, lambda);      
         
          [Test, ~] = test_computeLatentEstimation(Test, Model, lambda);      
          
          Test.fn_img_out = sprintf('%s/test%03d_result_noise%.4f_lambda%f.png', result_dir, idx_img, Test.sigma, lambda);
          imwrite(Test.ESTimg./Test.data_normalization_value, Test.fn_img_out);    
          psnr_output(idx_noise, idx_lambda, idx_img) = test_computePSNR(Test.ESTimg, Test.GT, Test.crop_width, Test.data_normalization_value);   
          %Test.fn_img_out = sprintf('%s/test%03d_result_noise%f_lambda%f_psnr%.4f.png', result_dir, idx_img, Test.sigma, lambda, psnr_output(idx_noise, idx_lambda, idx_img));
          %imwrite(Test.ESTimg./Test.data_normalization_value, Test.fn_img_out);      
          
          fprintf('\tprocess image %02d, lambda %f, psnr (%f, %f)\n', idx_img, lambda, psnr_input(idx_noise, idx_img), psnr_output(idx_noise, idx_lambda, idx_img));
          
    end
    end
end
mean(psnr_input(1,:))
mean(psnr_output(1,:))


