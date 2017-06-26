function [split_z_local] = test_eval_prox_denoise(cof, split_z_init, Test, Model)
 
layerscof = reshape(cof(1:Model.len_cof_shared), Model.len_layercof, Model.numDenoiseLayers); % discard lambda

% forward
% compute split_z^(i), phi^i(.), d_phi_i(.)
split_z_local = zeros(Test.imdims(1), Test.imdims(2), Model.numDenoiseLayers);
phi = zeros(Test.imdims(1), Test.imdims(2), Model.numFilters, Model.numDenoiseLayers);

for i = 1:1:Model.numDenoiseLayers  
    % initialize for current layer
    if(i>1)
        split_z_old = split_z_local(:,:,i-1);
    else
        split_z_old = split_z_init;
    end
    
    % compute filter component
    filterResponse = zeros(Model.numFilters, Test.imdims(1), Test.imdims(2));
    %par
    for j = 1:Model.numFilters
        fz = imfilter(split_z_old, Model.filters2D(:,:,j,i), 'circular', 'conv');
        lut_f = permute(Model.lut_f(i, j, :), [3 1 2]);
        [phi(:,:,j,i)] = test_evaluateInfluence(layerscof(:,i), j, fz, Model, Test.use_gpu, Test.use_lut, lut_f);
        filterResponse(j, :, :) = imfilter(phi(:,:,j,i), Model.filters2Dinv(:,:,j,i), 'circular','conv');
    end
    filterResponse = permute(sum(filterResponse, 1), [2 3 1]);
    
    %update split_z_local for current layer
    split_z_local(:,:,i) = split_z_old - filterResponse;    
     
end


end