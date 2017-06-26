function [Test, psnr] = test_computeLatentEstimation(Test, Model, lambda)
  

% compute for each training image
for i = 1:Test.N   
    
    y = Test.Meas(:, :, i);
    
    Test.x_init = Interpolation_Initial(y, ~Test.MtM);
    x = Test.x_init;
  
    split_z = zeros(Test.imdims(1), Test.imdims(2));
    split_z_local = zeros(Test.imdims(1), Test.imdims(2), Model.numDenoiseLayers);
       
    fprintf('inner iter ');
    for t = 2:Test.iter
         fprintf('%d', t-1);
         
        % compute rho value
        rho = Model.init_rho*Model.rho_ratio^(t-2);
                
        % update split_z
 	    split_z_init = x;

        [split_z_local(:,:,:)] = test_eval_prox_denoise(Model.cof, split_z_init, Test, Model); 
        
        split_z(:,:) = split_z_local(:,:,end);

        % update x        
         rhs = lambda.*Test.Mt.*y + rho.*split_z;
         lhs = lambda.*Test.MtM + rho;
         x_new = rhs./lhs;
         x(:,:) = x_new;
         
         psnr(t-1) = test_computePSNR(x_new, Test.GT, Test.crop_width, Test.data_normalization_value);
         fprintf(' (%f)\t', psnr(t-1));
        
    end
    
     % update
     Test.ESTimg(:, :, i) = x_new;      

 end


end