function [RBFweights] = initializeInfluence(Model, model_type, data_normalization_value)

% this function initialize the RBFs of the influence function rho_i.
if(strcmp(model_type, 'csf'))
    fn = sprintf('./initial_model/csf_rbf_weight_symmetric_%d.mat', Model.numRBFs);
else %diffusion
    fn = sprintf('./initial_model/diffusion_rbf_weight_symmetric_%d.mat', Model.numRBFs);
end
load(fn);

if(Model.use_symmetric_gmm)
    RBFweights = w(:); 
else
    RBFweights = [-flip(w); 0; w];
end

%-------------------------------------------------------------
if(strcmp(model_type, 'csf'))
    RBFweights = data_normalization_value.*RBFweights;
end
%--------------------------------------------------------------

end