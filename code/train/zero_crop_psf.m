function [out] = zero_crop_psf(in, kdims)

% crop the zerp-padded psf 'in' to be its original size kdims 
imdims = size(in);

if(mod(imdims(1), 2)==0)
      in = padarray(in, [1 0], 'post');
end
imh = size(in,1);
crop_region = (imh+1)/2 - (kdims(1)-1)/2 : (imh+1)/2 + (kdims(1)-1)/2;
out = in(crop_region, :);

in = out;

if(mod(imdims(2), 2)==0)
      in = padarray(in, [0 1], 'post');
end
imw = size(in,2);
crop_region = (imw+1)/2 - (kdims(2)-1)/2 : (imw+1)/2 + (kdims(2)-1)/2;
out = in(:, crop_region);



end