function [Test] = test_computeLatentEstimation(Test, Model, lambda)
  

% compute for each training image
for i = 1:Test.N   
    
    y = Test.Meas(:, :, i);
    fft_y = fft2(y);
    
    if(Test.wpsf>1)
        fft_k = psf2otf(Test.GTpsf_pad); % these can be pre-computed
        fft_kt = conj(fft_k);
        fft_ktk = abs(fft_k).^2;
        top_part_fidelity = lambda(i).*fft_kt.*fft_y;
        omega_part_fidelity = lambda(i).*fft_ktk;
        x =  y; % initialize the latent image to be the input image
    else
        top_part_fidelity = lambda(i).*fft_y;
        omega_part_fidelity = lambda(i).*ones(Test.imdims);
        x = y; % initialize the latent image to be the input image
    end

   
    fprintf('inner iter ');
    for t = 2:Test.iter
         fprintf('%d ', t-1);
         
        % compute rho value
        rho = Model.init_rho*Model.rho_ratio^(t-2);
    
        % compute omega
        omega = omega_part_fidelity + rho.*Model.fft_one; 
        
        % update split_z
        split_z = test_eval_prox_denoise(x, Model.lut_offset, Model.lut_step, Model.lut_f, Test.imdims, Model.filters2D, Model.filters2Dinv, Model.numDenoiseLayers, Model.numFilters); 
        
        % update x
        % compute new latent image estimate with current model parameter
        top =  top_part_fidelity + rho.*fft2(split_z(:,:));          
        
        x_new = max(0, real(ifft2(top./omega))); 
              
       x_new = fix_bndry(x_new(:), Test.GTpsf,  Test.mask, Test.PT, Test.imdims);
       x(:,:) = x_new;
      
       if(Test.save_intermediate)
           psnr = test_computePSNR(x, Test.GT, Test.crop_width, Test.data_normalization_value, Test.isLevinDataset);
           imwrite(x./255, sprintf('%s%d_psnr%.4f.png', Test.fn_img_interm, t, psnr));
           fprintf('  %.3f dB  ', psnr);
        end
         
    end
    
     % update
     Test.ESTimg(:, :, i) = max(0, min(Test.data_normalization_value, x_new)); 

 end


end