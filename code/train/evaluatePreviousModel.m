function [] = evaluatePreviousModel(cof, numlayers)

global Training Model Data
assert(Training.isPreTraining);
precompute_filters(cof);
layerscof = reshape(cof(1:Model.len_cof_shared), Model.len_layercof, Model.numDenoiseLayers); % discard lambda
MeaAll = cellfun(@(x) x.Meas, Data, 'UniformOutput', false);    
phi_init = zeros(Training.imdims(1), Training.imdims(2), Model.numFilters, Model.numDenoiseLayers);
Training_local = Training;
Model_local = Model;
Data_local = Data;

parfor idx = 1:Training.N
    b = MeaAll{idx}; %BE CAREFUL: only valid for the first splitting iteration where split_u = 0   
    phi = phi_init;
    split_z_local = zeros(size(b));
    for i = 1:numlayers    
        if(i>1)
            split_z_old = split_z_local;
        else
            split_z_old = b; % only for pre-training
        end
        
        % compute filter component
        filterResponse = zeros(Training_local.imdims);
        for j = 1:Model_local.numFilters
            fz = imfilter_warp(split_z_old, Model_local.filters2D(:,:,j,i));
            [phi(:,:,j,i), ~, ~] = evaluateInfluenceAndDerivative(layerscof(:,i), j, fz, Model_local, Training_local.use_gpu, Training_local.use_lut);
            filterResponse = filterResponse + imfilter_warp(phi(:,:,j,i), Model_local.filters2Dinv(:,:,j,i));
        end

        %update split_z_local for current layer
        split_z_local = split_z_old - filterResponse;    
    end
   
    Data_local{idx}.PreTrain.split_z_local = split_z_local;
end

Data = Data_local;


end