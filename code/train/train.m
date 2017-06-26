%------------------------------------------------
%------------------------------------------------
% training code for paper "discriminative transfer learning for general image restoration"
% contact: Lei Xiao (leixiao08@gmail.com)
% copyright reserved
%------------------------------------------------
%------------------------------------------------

clear all;  close all; clc;
fprintf('>>>start>>>\n');

% include path for all toolbox and folders
if ~isdeployed
   addpath(genpath('minFunc_2012'));
   addpath(genpath('subaxis'));
end

% set the number of cpu threads to use 
%********************************************************************************
M = 4;
reset_parpool(M);
%********************************************************************************

% create folder for results
global result_dir
result_dir = sprintf('result_%s',datestr(now, 'mmm_dd_HH_MM_SS'));
mkdir(result_dir);
fprintf('results are saved in this folder: %s\n', result_dir);

% reset random number generator for reproduce
reset(RandStream.getGlobalStream);

%% set-up
fprintf('\n--------------------------------------');
fprintf('\n------------Initialization--------------');
fprintf('\n--------------------------------------\n');

global Model Training Data 
Model = struct; % store the learned model
Training = struct; % stuff for training

%********************************************************************************
Training.learn_method = 'lbfgs'; 
Training.use_quantized_input = false; % set false for denoise comparison with certain previous methods
Training.use_gpu = true; % use GPU at certain functions that are evaluated extensively
Training.use_progressive_training = true; % progressively use more splitting iterations
if(1)
   Training.fn_trainingdata = '../../data/example_training_dataset/dataset400_CSF_discrete_denoise_5_5_25/dataset400_CSF_discrete_denoise_5_5_25.mat'; 
   Training.N =  400; % number of images used at training
else % only for check with small dataset
    Training.fn_trainingdata = '../../data/example_training_dataset/dataset5_CSF_discrete_denoise_5_5_25/dataset5_CSF_discrete_denoise_5_5_25.mat';
    Training.N =  5; % number of images used at training
end
Training.img_startIDX = 1; % index of the first image in the dataset
Training.num_task = 5;  fprintf('this code assumes each task contains the same number of training images.\n');
Training.num_iter_lbfgs_pretrain = 200; % number of LBFGS iterations at greedy pre-training
Training.num_iter_lbfgs_train = 100; % number of LBFGS iterations at training
%********************************************************************************
Model.numStages = 3; % number of splitting stages 
Model.numDenoiseLayers = 3; % number of denoising layers (within each splitting iteration)
Model.filterWidth = 5; % width of each 2D filter
%********************************************************************************

% Training structure
Training.use_pre_training = true; % only for the first splitting iteration
Training.use_lut = false; % disgard lut at training due to numerical accuracy issue
Training.task_type = 'ensemble'; % fixed
Training.method = 'bp'; % standard back-propagation (default)
Training.img_startIDX = Training.img_startIDX-1;
assert(Training.num_task<=Training.N);
assert(mod(Training.N, Training.num_task)==0);
nvis = min(5, Training.N); % number of images displayed for visualization
Training.magnify_ratio = 1e2; % magnify objective and gradient values (to prevent l-bfgs stopped too early)
Training.do_data_normalization = true; % whether to normalize all data
Training.data_normalization_value = 255; % the value for data normalization
Training.do_jpeg_compression = false; % fixed
Training.ref_noise_sigma = 25./255; % fixed
Training.idx_current_image = 0; % index of current image to be considered, temporary use
Training.iter_training = 0;
Training.idx_batch = 0;
Training.iter_alter = 0;
Training.isPreTraining = false;

% Model structure
Model.use_symmetric_gmm = true; % recommended
Model.model_type = 'diffusion'; 
Model.split_method = 'hqs'; 
Model.train_rho = false; % fixed
Model.use_separate_rho = false; % fixed as false for nonlinear-diffusion model
Model.use_separate_fidelity_weight = true; 
Model.use_noise_dep_fidelity_weight = true; % the fidelity weight of the image depends on noise level 
Model.init_rho = 1; %  weight for split quadratic term
Model.rho_ratio = 2; % increase rho after each hqs iteration
Model.init_weight_fidelity = 1; % initial value for data fidelity
Model.numStages = Model.numStages + 1; % index of valid iterations is 2, just for easier data storage
assert(Model.numStages>1);
assert(Model.filterWidth>=3);
Model.filter_type = 'rf'; % use random field as the filters in FoE model
Model.shrinkage_type = 'gmm'; 
Model.numFilters = (Model.filterWidth.^2 - 1); % number of 2D filters at each stage
Model.filterBetaSize = Model.filterWidth.^2 - 1; % use DCT basis removing DC term
Model.numRBFs = 63; % 63 used in TRND paper, 53 used in CSF paper
Model.numRBFs_reduced = (Model.numRBFs-1)/2;
if(Model.use_symmetric_gmm)
    Model.numRBFs_trained = Model.numRBFs_reduced;
else
    Model.numRBFs_trained = Model.numRBFs;
end
Model.len_layercof = Model.filterBetaSize*Model.numFilters + Model.numRBFs_trained*Model.numFilters; % length of cof's at each denoise layer
Model.len_cof_shared = Model.len_layercof *Model.numDenoiseLayers; 
if(Model.use_separate_fidelity_weight)
    Model.len_cof = Model.len_cof_shared + Training.N; % each trianing image has individual fidelity weight
else
    Model.len_cof = Model.len_cof_shared + 1; 
end
Model.cof = zeros(Model.len_cof, 1); 
% model coefficients: filters, and proximal operator, and scalar weights
Model.layercof_filters_startIDX = 1;
Model.layercof_RBFs_startIDX = Model.filterBetaSize*Model.numFilters + 1; 
Model.layercof_gdStep_IDX = Model.filterBetaSize*Model.numFilters + Model.numRBFs_trained*Model.numFilters + 1; 
Model.cof_fidelityWeight_IDX = Model.len_cof_shared + 1;
Model.stageIDX = 1;

% Data cell
Data = cell(Training.N, 1); % store the images 

%% set up optimization solver
% setup L-BFGS
fprintf('learn with lbfgs solver.\n');
Options = [];
Options.Method = 'lbfgs';%'sd'; %'csd'; %'bb';% %'cg';%'scg';% 'pcg';% 'lbfgs';
Options.Display = '(iter)'; %'full';
Options.optTol = 1e-8;
Options.progTol = 1e-8;    
Training.fval_trace = cell(1, Model.numStages -1);

%% initialization
loadTrainingData();
initializeModelCoefficients();
initializeCliques(); % initialize clique matrix for easier computation       
visualizeFilters();
visualizeInfluenceFunctions(); 
save(sprintf('%s/workspace.mat', result_dir), 'Training', 'Model', 'Data', 'result_dir');
modelcof = Model.cof;
save(sprintf('%s/modelcof_init.mat', result_dir), 'modelcof');

%% start pre-training  for the first splitting iteration
if(Training.use_pre_training)    
    % only need when use multi-layer denoiser
    if(Model.numDenoiseLayers>1) 
        fprintf('\n--------------------------------------');
        fprintf('\n---------PRE-TRAINING-----------');
        fprintf('\n--------------------------------------\n');
        tic 

        Training.isPreTraining = true;
        Training.iter_training = 2; 

        % first greedy training for each layer
        for layer = 1:Model.numDenoiseLayers
            if(isdeployed)
                reset_parpool(M); % reset parpool in case of connection issue on the cluster
            end
            fprintf('pre-training %d-th layer.\n', layer);
            Training.PreTrain.layer = layer;

            % evaluate previous model and save the output which is used as input in next pretraining.
            if(layer>1)
               evaluatePreviousModel(Model.cof, layer-1); 
            end

           cof_in = Model.cof(:);      
              
           Options.MaxIter = Training.num_iter_lbfgs_pretrain;
           Training.maxIter = Options.MaxIter; % temporary
           handle_loss_func = @(cof) loss_func(cof, Training, Model, Data);  
           [cof_out, ~, exitflag, output] = minFunc(handle_loss_func, cof_in, Options);           
           Training.fval_trace_pretrain = output.trace.fval(:);

            evaluateModel(cof_out, nvis);

            [psnr_all_pre, psnr_mean_pre(layer)] = computePSNR();
            fprintf('pre-training, mean psnr %f dB.\n', psnr_mean_pre(layer));
            vis_task_psnr(psnr_all_pre);
            save(sprintf('%s/workspace.mat', result_dir), 'Training', 'Model', 'Data', 'result_dir');
            modelcof = Model.cof;
            save(sprintf('%s/modelcof_pretrain_%d_layers.mat', result_dir, layer), 'modelcof');
        end

        % then jointly train all layers (if use progressive training later on, this step can be skiped)
        fprintf('skip joint pre-training as we assume to use progressive training later on.\n');

        time_pretrain = toc;
        fprintf('time pre-training = %f sec', time_pretrain);
    end

    Training.isPreTraining = false; 
end

%% continue training
fprintf('\n--------------------------------------');
fprintf('\n------------TRAINING--------------');
fprintf('\n--------------------------------------\n');
tic

if(Training.use_progressive_training)
    t_start = 2;
    if(Model.numStages<5||strcmp(Model.split_method, 'hqs'))
        t_step = 1;
    else
        t_step = 2; % increase two iterations progressively
    end
else
    t_start = Model.numStages;
    t_step = 1;
end

for t = t_start:t_step:Model.numStages
    if(isdeployed)
        reset_parpool(M); % reset parpool in case of connection issue on the cluster
    end
    fprintf('--training, stage %d out of %d.\n', t, Model.numStages);
    Training.iter_training = t;
              
    cof_in = Model.cof(:);
       
    Options.MaxIter = Training.num_iter_lbfgs_train;
    Training.maxIter = Options.MaxIter; % temporary      
    handle_loss_func = @(cof) loss_func(cof, Training, Model, Data);
    [cof_out, ~, exitflag, output] = minFunc(handle_loss_func, cof_in, Options);
    Training.fval_trace{t-1} = output.trace.fval(:);    
   
    % update model coef
    evaluateModel(cof_out, nvis);
    
    [psnr_all, psnr_mean] = computePSNR();
    fprintf('mean psnr %f dB.\n',  psnr_mean);
    vis_task_psnr(psnr_all);
    %visualizeEstimationAll(nvis);
    save(sprintf('%s/workspace.mat', result_dir), 'Training', 'Model', 'Data', 'result_dir');
    modelcof = Model.cof;
    save(sprintf('%s/modelcof_%d_layers_iter_%d.mat', result_dir, Model.numDenoiseLayers, t), 'modelcof');
end


fprintf('\ntraining done.\n');
time_training = toc;
fprintf('time training = %f sec.\n', time_training);
