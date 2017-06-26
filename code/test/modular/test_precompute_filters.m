function  [Model] = test_precompute_filters(Test, Model, cof)

% precompute filters from the given model coefficients 'cof'

% set to zero
Model.filters2D = 0.*Model.filters2D;
Model.filters2Dinv = 0.*Model.filters2Dinv;

for layer = 1:Model.numDenoiseLayers
    for i = 1:Model.numFilters    
        idx_start = (i-1)*Model.filterBetaSize + Model.layercof_filters_startIDX + (layer-1)*Model.len_layercof;
        idx_end = idx_start+Model.filterBetaSize-1;
        cof_beta = cof(idx_start : idx_end);
        tmpf = Model.Basis * cof_beta;
        filter = reshape(tmpf./norm(tmpf), Model.filterWidth, Model.filterWidth);      
        filter_inv = rot90(rot90(filter));    
        Model.filters2D(:, :, i, layer) = filter;
        Model.filters2Dinv(:, :, i, layer) = filter_inv;   
    end
end
    
Model.fft_one = ones(Test.imdims);

end