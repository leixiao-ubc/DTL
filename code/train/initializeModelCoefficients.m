function  [] = initializeModelCoefficients()

global Training Model Data

% weight of data fidelity
noise_sigma = cellfun(@(x) x.noise_sigma, Data, 'UniformOutput', true);
Training.relative_noise_variance = (Training.ref_noise_sigma./noise_sigma).^2;

if(Model.use_separate_fidelity_weight)
    lambda = expident_inv(Model.init_weight_fidelity.*Training.relative_noise_variance); 
else
    lambda = expident_inv(Model.init_weight_fidelity);
end

cof_init = zeros(Model.len_cof, 1);

%-------------------------------------------
%these values are taken from TRD paper
scale_RBFweight = [10, 5, 1, 1, 1];
assert(Model.numDenoiseLayers<6);
%-------------------------------------------

for layer = 1:Model.numDenoiseLayers
    layercof_init = zeros(Model.len_layercof, 1);
    sw = scale_RBFweight(layer); 
    
    % filters
    if(strcmp(Model.filter_type, 'rf'))
        % filter is better to be normalized and have zero-mean
        % use DCT basis as the basis
        m = Model.filterWidth^2;
        C = zeros(m, m-1);
        for i = 2:m
                a = zeros(Model.filterWidth,Model.filterWidth);
                a(i) = 1;
                b = idct2(a');
                C(:, i-1) = b(:);
        end
        Model.Basis = C(:, 1:end); 
        Model.Basis_t = Model.Basis';
        Model.Basis_t_Basis = Model.Basis_t*Model.Basis;
        b = eye(m-1);
        for i = 1:Model.numFilters
            idx = (i-1)*(m-1) + Model.layercof_filters_startIDX - 1; 
            layercof_init(idx+1 : idx+m-1) = b(:, i);
        end
    end    

    % proximal operator
    if(strcmp(Model.shrinkage_type, 'gmm'))
        support = Training.data_normalization_value./255.*[-310, 310]; % the range of the centers (mu) of these RBFs
        interval = (support(2) - support(1))/(Model.numRBFs - 1); 
        Model.RBFcenters = support(1):interval:support(2);
        Model.RBFscale = interval; 
        if(Model.use_symmetric_gmm)
          RBFweights_reduced = sw.*initializeInfluence(Model, Model.model_type, Training.data_normalization_value); 
          layercof_init(Model.layercof_RBFs_startIDX: Model.layercof_RBFs_startIDX + Model.numRBFs_reduced*Model.numFilters - 1) = repmat(RBFweights_reduced, [1, Model.numFilters]); 
        else
          RBFweights = sw.*initializeInfluence(Model, Model.model_type, Training.data_normalization_value);  %TRND: 10, 5, 1, 1, ...
          layercof_init(Model.layercof_RBFs_startIDX: Model.layercof_RBFs_startIDX + Model.numRBFs*Model.numFilters - 1) = repmat(RBFweights, [1, Model.numFilters]); 
        end
    %elseif(strcmp(Model.shrinkage_type, 'soft')) % set the penality function as L1 norm, and thus the shrinkage function is soft-thresholding     
         %layercof_init(Model.layercof_RBFs_startIDX: Model.layercof_RBFs_startIDX + Model.numRBFs_reduced*Model.numFilters - 1) = repmat(tau, [1, Model.numFilters]); 
    end

    offset = (layer-1)*Model.len_layercof;
    cof_init(offset+1:offset+Model.len_layercof) = layercof_init(:);
end

% repmat layercof 
cof_init(1:Model.len_cof_shared) = repmat(layercof_init(:), [Model.numDenoiseLayers, 1]);

% fidelity weight
if(Model.use_separate_fidelity_weight)
    cof_init(Model.cof_fidelityWeight_IDX:Model.cof_fidelityWeight_IDX+Training.N-1) = lambda(:); % weight of data-fidelity
else
    cof_init(Model.cof_fidelityWeight_IDX) = lambda;
end

Model.cof = cof_init;

% pre-allocate
Model.filters2D = zeros(Model.filterWidth, Model.filterWidth, Model.numFilters, Model.numDenoiseLayers);
Model.filters2Dinv = zeros(Model.filterWidth, Model.filterWidth, Model.numFilters, Model.numDenoiseLayers);
Model.fft_one = zeros(Training.imdims); 

fprintf('models initialized.\n');
       
end
