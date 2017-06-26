function [] = computeLatentEstimation()

global Training Model Data

% lambda
if(Model.use_separate_fidelity_weight)    
    tmp = expident(Model.cof(Model.cof_fidelityWeight_IDX:Model.cof_fidelityWeight_IDX+Training.N-1)); % weight of fidelity term
elseif(Model.use_linear_fidelity_weight)
    tmp = expident(Model.cof(Model.cof_fidelityWeight_IDX).*Training.relative_noise_variance);
else
    tmp = expident(repmat(Model.cof(Model.cof_fidelityWeight_IDX), [Training.N, 1])); % weight of fidelity term
end
lambda = tmp.value;

cof = Model.cof;

if(Training.isPreTraining)
    T = Training.PreTrain.layer;
else
    T = Model.numDenoiseLayers;
end

% compute for each training image
Data_local = Data;

parfor i = 1:Training.N
    y = Data_local{i}.Meas;    
    fft_y = fft2(y);    
    if(Data_local{i}.valid_psf_width>1)
        k = Data_local{i}.GTpsf_pad;
        fft_k = Data_local{i}.fft_k;
        fft_kt = Data_local{i}.fft_kt;
        fft_ktk = Data_local{i}.fft_ktk;
        top_part_fidelity = lambda(i).*fft_kt.*fft_y;
        omega_part_fidelity = lambda(i).*fft_ktk;
        x =  y; % initialize the latent image to be the input image
    else
        top_part_fidelity = lambda(i).*fft_y;
        omega_part_fidelity = lambda(i).*ones(Training.imdims);
        x = y; % initialize the latent image to be the input image
    end
    
    split_z = zeros(Training.imdims(1), Training.imdims(2), Training.iter_training); % store Prox(x^(t-1))
    split_z_local = zeros(Training.imdims(1), Training.imdims(2), Model.numDenoiseLayers,Training.iter_training);

     for t = 2:Training.iter_training         
        % compute rho value
        rho = Model.init_rho*Model.rho_ratio^(t-2);
    
        % compute omega
        omega = omega_part_fidelity + rho.*Model.fft_one; 
        
        % update split_z, split_u
        split_z_init = x(:,:,t-1);
        [split_z_local(:,:,:,t)] = eval_prox_denoise(cof, split_z_init, Training, Model);         
        split_z(:,:,t) = split_z_local(:,:,T,t);

        % update x
         top =  top_part_fidelity + rho.*fft2(split_z(:,:,t));          
         x(:,:,t) = real(ifft2(top./omega));                   
     end
    
     % update estimate
     Data_local{i}.ESTimg = x(:,:,end);          
end
 
Data = Data_local; 

end
