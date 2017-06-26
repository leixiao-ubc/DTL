function [Test, runtime] = test_computeLatentEstimation(Test, Model, lambda)

for i = 1:Test.N   
    
    y = Test.Meas(:, :, i);   
    x = y; % initialize the latent image to be the input image
      
    tic
    for t = 2:Test.iter   
        rho = Model.init_rho*Model.rho_ratio^(t-2);    
        split_z = test_eval_prox_denoise(x, Model.lut_offset, Model.lut_step, Model.lut_f, Test.imdims, Model.filters2D, Model.filters2Dinv, Model.numDenoiseLayers, Model.numFilters); 
        x_new =  (lambda.*y + rho.*split_z)./(lambda + rho);
        x = x_new;      
        
        if(Test.save_intermediate)
           psnr = test_computePSNR(x, Test.GT, Test.crop_width, Test.data_normalization_value);
           imwrite(x./255, sprintf('%s%d_psnr%.4f.png', Test.fn_img_interm, t, psnr));
        end
    end
    runtime = toc;
    
    Test.ESTimg(:, :, i) = max(0, min(x_new, Test.data_normalization_value));
 end


end