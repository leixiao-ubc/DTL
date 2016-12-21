function [f, df, c] = evaluateInfluenceAndDerivative(cof, idx, in, Model, use_gpu, use_lut, idx_img)
% apply the influence functions (1D) 

% INPUT:
% idx: index of the influence function to evaluate
% in: matrix w/ dimension of imgHeight x imgWidth x numTrainingImgs

% OUTPUT:
% f: influence, matrix with the same dimension as the input in
% df: first order derivative of the influence function, matrix with the same dimension as the input in

size_in = size(in);
x = (in(:)'); % row vector    
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
     c = a(Model.numRBFs_reduced+1:end, :) - a(Model.numRBFs_reduced:-1:1, :);  % used outside of this function
     f = bsxfun(@times, a, weight);
     df = bsxfun(@times, f, x_minus_mu); 
     f = reshape(sum(f, 1), size_in);
     df = reshape(-2.*rbf_scale_constant.*sum(df, 1), size_in);     
     if(use_gpu)
         f = gather(f);
         df = gather(df);
         c = gather(c);
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
     df = bsxfun(@times, f, x_minus_mu); 
     f = reshape(sum(f, 1), size_in);
     df = reshape(-2.*rbf_scale_constant.*sum(df, 1), size_in);     
     c = a;
     if(use_gpu)
         f = gather(f);
         df = gather(df);
         c = gather(c);
     end
end
   
  
end









