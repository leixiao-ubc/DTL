function [Test] = test_loadData(Test, Model, Data)

Test.N = 1;
Test.GTpsf = Data.GTpsf;

% pre-allocate memory and pre-compute some matrices
Test.kdims = size(Test.GTpsf);
Test.wpsf = Test.kdims(1);
if(Test.prepad)
   Test.wpad = max((Test.wpsf-1)/2, (Model.filterWidth-1)/2); 
else
   Test.wpad = 0;
end

img_gt = Test.data_normalization_value.*Data.GTimg;


Test.gtimdims = size(img_gt);
hi = Test.gtimdims(1); wi = Test.gtimdims(2); % before padding
Test.imdims = [hi + Test.wpad*2, wi + Test.wpad*2];
Test.Meas = zeros(hi + Test.wpad*2, wi + Test.wpad*2, Test.N);
Test.GT = zeros(hi + Test.wpad*2, wi + Test.wpad*2, Test.N);
Test.GTpsf_pad = zero_pad_psf(Test.GTpsf, Test.imdims);

% precompute truncating and padding matrix (code borrowed from CSF paper)
npixels = prod(Test.imdims);
t = Test.wpad; 
[r,c] = ndgrid(1+t:Test.imdims(1)-t, 1+t:Test.imdims(2)-t);
ind_int = sub2ind(Test.imdims, r(:), c(:));
d = zeros(Test.imdims); d(ind_int) = 1;
T = spdiags(d(:),0,npixels,npixels);
T = T(ind_int,:);
idximg = reshape(1:prod(Test.imdims-2*t),Test.imdims-2*t);
pad_idximg = padarray(idximg,[t t],'replicate','both');
P = sparse((1:npixels)',pad_idximg(:),ones(npixels,1),npixels,prod(Test.imdims-2*t));
PT = P*T; % first truncation, then padding
Test.T = T;
Test.PT = PT;

img_used = Test.data_normalization_value.*Data.Meas; 
img_used = padarray(img_used, Test.wpad.*[1,1], 'replicate', 'both');  
Test.mask = edgetaper_mask(Test.GTpsf, Test.imdims);
img_used = fix_bndry(img_used(:), Test.GTpsf,  Test.mask, Test.PT, Test.imdims);

if(Test.use_quantized_meas)
    img_used = uint8(max(0, min(Test.data_normalization_value, img_used)));
end
img_used = double(img_used);

Test.Meas = img_used;
Test.GT = reshape(Test.PT*Test.T'*(img_gt(:)), Test.imdims);
Test.ESTimg = [];


end