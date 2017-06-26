function [split_z] = test_eval_prox_denoise(split_z_init, lut_offset, lut_step, lut_f, imdims, filters2D, filters2Dinv, numDenoiseLayers, numFilters)

split_z = split_z_init;

for i = 1:numDenoiseLayers  
      
    % compute filter component
    filterResponse = zeros(imdims);
    parfor j = 1:numFilters
        fz = imfilter(split_z, filters2D(:,:,j,i), 'circular', 'conv');
        phi = test_evaluateInfluence(fz, lut_offset, lut_step, lut_f(:,i,j));
        filterResponse = filterResponse + imfilter(phi, filters2Dinv(:,:,j,i), 'circular', 'conv');
    end
    split_z = split_z - filterResponse;    
     
end


end