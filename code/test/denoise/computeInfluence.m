function [f] = computeInfluence(cof, idx, in, Model, use_gpu, use_lut, lut_f)
% apply the influence functions (1D) 

size_in = size(in);
x = (in(:)'); % row vector    

if(strcmp(Model.shrinkage_type, 'gmm'))
    
    if(use_lut&&Model.lut_exist)
        [f] = lut_eval(x, Model.lut_offset, Model.lut_step, lut_f, 0, 0, 0);  
        f = reshape(f, size_in);
    else        
        rbf_scale_constant = 1/(2*Model.RBFscale^2);
        if(Model.use_symmetric_gmm)
            start = Model.layercof_RBFs_startIDX - 1 + (idx - 1)*Model.numRBFs_reduced;     
             weight = cof(start+1:start+Model.numRBFs_reduced); % vector
             mu =  Model.RBFcenters(Model.numRBFs_reduced+2:Model.numRBFs_reduced*2+1); % vector
             weight = (weight(:)); % column vector    
             mu = (mu(:)); % column vector            
             if(use_gpu)
                x = gpuArray(x); % row vector    
                weight = gpuArray(weight);   
                mu = gpuArray(mu);  
             end      
             weight = [-flipud(weight); weight];
             mu = [-flipud(mu); mu];  
             x_minus_mu = bsxfun(@minus, x, mu); 
             a = exp(-rbf_scale_constant.*x_minus_mu.^2);
             f = bsxfun(@times, a, weight);
             f = reshape(sum(f, 1), size_in);
             if(use_gpu)
                 f = gather(f);
             end
        else % none symmetric constraint
             start = Model.layercof_RBFs_startIDX - 1 + (idx - 1)*Model.numRBFs;     
             weight = cof(start+1:start+Model.numRBFs); % vector
             mu =  Model.RBFcenters(:); % vector
             weight = (weight(:)); % column vector    
             mu = (mu(:)); % column vector            
             if(use_gpu)
                x = gpuArray(x); % row vector    
                weight = gpuArray(weight);   
                mu = gpuArray(mu);  
             end      
             x_minus_mu = bsxfun(@minus, x, mu); 
             a = exp(-rbf_scale_constant.*x_minus_mu.^2);
             f = bsxfun(@times, a, weight);
             f = reshape(sum(f, 1), size_in);
             if(use_gpu)
                 f = gather(f);
             end
        end
    
    end     
   
elseif(strcmp(Model.shrinkage_type, 'soft'))
     % removed
end
  
end









