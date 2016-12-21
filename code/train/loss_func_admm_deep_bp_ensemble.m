function [f, g] = loss_func_admm_deep_bp_ensemble(cof, Training, Model, DataOne) 
% evaluate the objective and gradient of the loss func 
% for SINGLE image

%use half-quadratic-splitting

% INPUT:
% cof: model coefficients which are to be evaluated inside minFunc solver

% OUTPUT:
% f: objective 
% g: gradient of the objective to each coefficient

%global Model Training

f = 0; 
g = zeros(size(cof)); % set objective and gradient to zero

rho_array = zeros(Training.iter_training, 1); %+20161031
omega_array = zeros(Training.imdims(1), Training.imdims(2), Training.iter_training); %+20161031

tmp = expident(cof(end)); % weight of fidelity term
lambda = tmp.value;
lambda_grad = tmp.grad;

rs = @(z) reshape(z, Training.imdims); % function handle to reshape (untruncated) image from vector
rst = @(z) reshape(z, Training.gtimdims); % function handle to reshape (truncated) image from vector

len_x = prod(Training.imdims);

%variables stored at forward iterations, and reused at backpropagation::
x = ones(Training.imdims(1), Training.imdims(2), Training.iter_training);
split_z = ones(Training.imdims(1), Training.imdims(2), Training.iter_training); % store Prox(x^(t-1))
split_z_local = ones(Training.imdims(1), Training.imdims(2), Model.numDenoiseLayers,Training.iter_training);
phi = ones(Training.imdims(1), Training.imdims(2), Model.numFilters, Model.numDenoiseLayers,Training.iter_training);
d_phi = ones(Training.imdims(1), Training.imdims(2), Model.numFilters, Model.numDenoiseLayers,Training.iter_training);
c = ones(Model.numRBFs_trained, prod(Training.imdims), Model.numFilters, Model.numDenoiseLayers,Training.iter_training);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   compute objective   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 y = DataOne.Meas;
 fft_y = fft2(y);

if(DataOne.valid_psf_width>1)
    fft_k = DataOne.fft_k;
    fft_kt = DataOne.fft_kt;
    fft_ktk = DataOne.fft_ktk;
    top_part_fidelity = lambda.*fft_kt.*fft_y;
    omega_part_fidelity = lambda.*fft_ktk;
    x(:,:,1) = y; % initialize the latent image to be the input image
else
    top_part_fidelity = lambda.*fft_y;
    omega_part_fidelity = lambda.*ones(Training.imdims);
    x(:,:,1) = y; % initialize the latent image to be the input image
end

% for remaining iterations (where the model is actually used)
split_z(:,:,1) = x(:,:,1); % initialization   

 for t = 2:Training.iter_training     
    % compute rho value
    rho_array(t) = Model.init_rho*Model.rho_ratio^(t-2);
     
    % compute omega (as rho value changes)
    omega_array(:,:,t) = omega_part_fidelity + rho_array(t).*Model.fft_one; 
     
    split_z_init = x(:,:,t-1);
    
    [split_z_local(:,:,:,t), phi(:,:,:,:,t), d_phi(:,:,:,:,t), c(:,:,:,:,t)] = prox_denoise(cof, split_z_init, Training, Model); 
        
    split_z(:,:,t) = split_z_local(:,:,end,t);

    % update x
    % compute new latent image estimate with current model parameter
     top =  top_part_fidelity + rho_array(t).*fft2(split_z(:,:,t));% + split_u(:,:,t)); 
     x_new = real(ifft2(top./omega_array(:,:,t))); 
     x(:,:,t) = x_new;
 end

% compute objective
residual_x = rst(Training.T*reshape(x(:,:,end), [len_x, 1])) - DataOne.GTimg;
residual_x = rs(Training.T'*residual_x(:));    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   compute gradient using back-propagation   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mse = sumsqr(residual_x)./prod(Training.gtimdims);  
f = -20*log10(Training.data_normalization_value) + 10*log10(mse);
g_x =   (20/(log(10)*mse*prod(Training.gtimdims))) .* residual_x;   

g_lambda = 0;
g_cof_denoise = zeros(Model.len_cof_shared, 1);

for t = Training.iter_training:-1:2     
    % gradient g >> x^(t) >> alpha  [to accumulate]
    % gradient g >> x^(t) >> lambda  [to accumulate]
     % gradient g >> x^(t) >> f_i      [to accumulate]
     % gradient g >> x^(t) >> RBF_i      [to accumulate]
     % gradient g >> x^(t) >> x^(t-1)      [to reuse]
     % gradient g >> x^(t) >> split_u^(t-1)       [to reuse]    
     % gradient g >> split_u^(t) >> alpha  [to accumulate]
     % gradient g >> split_u^(t) >> f_i      [to accumulate]
     % gradient g >> split_u^(t) >> RBF_i      [to accumulate]
     % gradient g >> split_u^(t) >> x^(t-1)       [to reuse]
     % gradient g >> split_u^(t) >> split_u^(t-1)      [to accumulate]               
     g_x_old = 0.*g_x;

     a = real(ifft2(fft2(g_x)./omega_array(:,:,t))); % (Omega^-1) * (g_x) [correspond to 'u' in my draft note]
     x_new = x(:,:,t);
     fft_x_new = fft2(x_new);

     % gradient g >> x^(t) >> lambda  [to accumulate]
     if(DataOne.valid_psf_width>1)             
         x_lambda = real(ifft2(fft_kt.*(fft_y - fft_k.*fft_x_new)));
     else
         x_lambda =  y - x_new;
     end
     g_lambda = g_lambda + lambda_grad.*dot(x_lambda(:), a(:));

     % compute gradient g>>...>> model parameters (f_i, RBF_i, alpha)
     split_z_init = x(:,:,t-1);    

     [db, tp3] = gradient_prox_denoise(cof, split_z_init, a, rho_array(t), Training, Model, split_z_local(:,:,:,t), phi(:,:,:,:,t), d_phi(:,:,:,:,t), c(:,:,:,:,t));
     g_cof_denoise = g_cof_denoise + tp3;     
  
     % gradient g >> x^(t) >> x^(t-1)   [to reuse]
     g_x_old = g_x_old + db;
     
     g_x = g_x_old; % update gradient g >> .. >> x^(t-1) for next iteration    
end

g = g + [g_cof_denoise(:); g_lambda(:)];



end