function [Test, psnr] = test_computeLatentEstimation(Test, Model, lambda)
  

% compute for each training image
for i = 1:Test.N   
    
    y = Test.Meas(:, :, i);
    fft_y = fft2(y);
    
    top_part_fidelity = lambda(i).*fft_y;
    omega_part_fidelity = lambda(i).*ones(Test.imdims);
    x = y; % initialize the latent image to be the input image

    
    split_z = zeros(Test.imdims(1), Test.imdims(2));
    split_z_local = zeros(Test.imdims(1), Test.imdims(2), Model.numDenoiseLayers);
   
    
    fprintf('inner iter ');
    for t = 2:Test.iter
         fprintf('%d', t-1);
         
        % compute rho value
        rho = Model.init_rho*Model.rho_ratio^(t-2);
        rho_bm3d = Test.init_rho_bm3d*Test.rho_ratio_bm3d^(t-2);
    
        % compute omega
        omega = omega_part_fidelity + (rho + rho_bm3d).*Model.fft_one; 
        
        % update split_z
 	    split_z_init = x;

        [split_z_local(:,:,:)] = test_eval_prox_denoise(Model.cof, split_z_init, Test, Model); 
        
        split_z(:,:) = split_z_local(:,:,end);

        % update split_v (run BM3D)
        split_v = runBM3D(x, Test.lambda_bm3d./rho_bm3d, Test.data_normalization_value);        
                
        % update x
        % compute new latent image estimate with current model parameter
         top =  top_part_fidelity + rho.*fft2(split_z(:,:)) + rho_bm3d.*fft2(split_v(:,:));          
         x_new = real(ifft2(top./omega)); 
         x(:,:) = x_new;
         
         psnr(t-1) = test_computePSNR(x_new, Test.GT, Test.crop_width, Test.data_normalization_value);
         fprintf(' (%f)\t', psnr(t-1));
         
    end
    
     % update
     Test.ESTimg(:, :, i) = x_new;      

    
 end


end