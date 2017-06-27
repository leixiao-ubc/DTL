function [Test, psnr] = test_computeLatentEstimation(Test, Model, lambda)
  
psnr = 0;

% compute for each training image
for i = 1:Test.N   
    
    y = Test.Meas(:, :, i);
    
    Test.x_init = Interpolation_Initial(y, ~Test.MtM);
    x = Test.x_init;
         
    fprintf('inner iter ');
    for t = 2:Test.iter
         fprintf('%d', t-1);
         
        % compute rho value
        rho = Model.init_rho*Model.rho_ratio^(t-2);
                
        % update split_z
        split_z = test_eval_prox_denoise(x, Model.lut_offset, Model.lut_step, Model.lut_f, Test.imdims, Model.filters2D, Model.filters2Dinv, Model.numDenoiseLayers, Model.numFilters); 
        
        % update x        
         rhs = lambda.*Test.Mt.*y + rho.*split_z;
         lhs = lambda.*Test.MtM + rho;
         x_new = rhs./lhs;
         x(:,:) = x_new;
         
         if(Test.save_intermediate)
            psnr = test_computePSNR(x_new, Test.GT, Test.crop_width, Test.data_normalization_value);
            fprintf(' (%f)\t', psnr);
         end
    end
    
     % update
     Test.ESTimg(:, :, i) = x_new;      

 end


end