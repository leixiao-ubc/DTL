function [out] = zero_pad_psf(in, imdims)

% zero pad the psf 'in' to be size imdims
wpsf = size(in, 1);   

if(mod(imdims(1), 2)==0)
     padsize = (imdims(1) + 1 - wpsf)/2;
     tmp = padarray(in, [padsize 0], 'both');
     out = tmp(1:end-1, :);  
else
     padsize = (imdims(1) - wpsf)/2;
     out = padarray(in, [padsize 0], 'both');  
end

in = out;

if(mod(imdims(2), 2)==0)
     padsize = (imdims(2) + 1 - wpsf)/2;
     tmp = padarray(in, [0 padsize], 'both');
     out = tmp(:, 1:end-1);  
else
     padsize = (imdims(2) - wpsf)/2;
     out = padarray(in, [0 padsize], 'both');  
end


end