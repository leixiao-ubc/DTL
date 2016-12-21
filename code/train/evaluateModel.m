function [] =  evaluateModel(cof_out, nvis)

global Model 

Model.cof(:)  = cof_out(:); % update the model coefficients 
precompute_filters(Model.cof(:));
visualizeFilters();
visualizeInfluenceFunctions(); 

% compute latent image given current model and numStages
computeLatentEstimation(); 
visualizeEstimation(nvis);     
        
end