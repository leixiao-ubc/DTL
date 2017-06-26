function [f, g] = loss_func(cof, Training, Model, Data)

% evaluate the objective and gradient of the loss func 
% INPUT:
% cof: model coefficients which are to be evaluated inside minFunc solver


% OUTPUT:
% f: objective 
% g: gradient of the objective to each coefficient

f = 0; 
g = zeros(size(cof)); % set objective and gradient to zero

% pre-compute 2d filters, and corresponding Fourier transform (always use convolution for filtering)
[Model] = precompute_filters_at_learning(cof, Model, Training.imdims);

% fetch current coefficients
cof_single = zeros(Training.N, Model.len_cof_shared+1); % shared part + fidelity weight
cof_single(:, 1:Model.len_cof_shared) = repmat(cof(1:Model.len_cof_shared)', [Training.N, 1]);
if(Model.use_separate_fidelity_weight)
   cof_single(:, end) = cof(Model.len_cof_shared+1:end); % fidelity weight lambda for each image
else
   cof_single(:, end) = cof(Model.len_cof_shared+1:end);
end

g_single = zeros(size(cof_single));
f_single = zeros(Training.N, 1);

%------------------------------------------------------------------------------------------------------
isPreTraining = Training.isPreTraining;
parfor i = 1:Training.N
   tp = cof_single(i, :);
   if(isPreTraining)
       [f_single(i), g_single(i, :)] = loss_func_hqs_deep_bp_ensemble_pretrain(tp(:), Training, Model, Data{i}); 
   else
       [f_single(i), g_single(i, :)] = loss_func_hqs_deep_bp_ensemble(tp(:), Training, Model, Data{i}); 
   end
end
%------------------------------------------------------------------------------------------------------

f = sum(f_single, 1); % scalar
g(1:Model.len_cof_shared) =sum(g_single(:, 1:Model.len_cof_shared), 1); %vector

if(Model.use_separate_fidelity_weight)
    if(Model.use_noise_dep_fidelity_weight) % depends on noise level
       g_lambda_tmp = reshape(g_single(:, end), [Training.N/Training.num_task, Training.num_task]);  % dirty :: assume each task contains the same number of images
       g_lambda = reshape(repmat(mean(g_lambda_tmp, 1), [Training.N/Training.num_task, 1]), [Training.N, 1]);
       g(Model.len_cof_shared +1:end) =  g_lambda(:);
    else % totally separate
        g(Model.len_cof_shared +1:end) = g_single(:, end);
    end
else 
    g(Model.len_cof_shared +1) = sum(g_single(:, end)); %
end
    
f = f./Training.N.*Training.magnify_ratio;
g = g./Training.N.*Training.magnify_ratio;

fprintf('.');    

    
end
