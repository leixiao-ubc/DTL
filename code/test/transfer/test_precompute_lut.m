function  [Model] = test_precompute_lut(cof, Model, use_gpu)

Model.lut_offset = -310;
Model.lut_step = 0.05;

in = [Model.lut_offset : Model.lut_step : -Model.lut_offset]; % precomputed range
len_in = length(in);
lut_f = zeros(Model.numDenoiseLayers, Model.numFilters, len_in);

for layer = 1:Model.numDenoiseLayers
    idx_start = (layer-1)*Model.len_layercof + 1;
    idx_end = idx_start + Model.len_layercof - 1;
    cof_current = cof(idx_start:idx_end);
    for j = 1:Model.numFilters
        [f] = test_evaluateInfluence(cof_current, j, in, Model, use_gpu, false); 
        lut_f(layer, j, :) = f(:);
    end
end

Model.lut_f = lut_f;
Model.lut_exist = true;
fprintf('look-up-table precomputed.\n');
    
end