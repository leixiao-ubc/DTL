function [Test] = test_loadData(Test, Model)

%%
Test.N = 1;

% load test image
img_gt = imread(Test.fn_img_gt);
if(size(img_gt,3)==3)
    img_gt = rgb2gray(img_gt);
end
img_gt = double(img_gt);

% pre-allocate memory and pre-compute some matrices
Test.wpsf = Test.kdims(1);
Test.wpad = 0;% max((Test.wpsf-1)/2, (Model.filterWidth-1)/2); 
Test.gtimdims = size(img_gt);
hi = Test.gtimdims(1); wi = Test.gtimdims(2); % before padding
Test.imdims = [hi + Test.wpad*2, wi + Test.wpad*2];
Test.Meas = zeros(hi + Test.wpad*2, wi + Test.wpad*2, Test.N);
Test.GT = zeros(hi + Test.wpad*2, wi + Test.wpad*2, Test.N);

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

%-------------------------------------------
%-------------------------------------------
img_used = img_gt + Test.sigma.*randn(size(img_gt));

if(Test.use_quantized_meas)
    img_used = uint8(max(0, min(Test.data_normalization_value, img_used)));
end
img_used = double(img_used);
%-------------------------------------------
%-------------------------------------------

Test.Meas = reshape(Test.PT*(Test.T'*img_used(:)), Test.imdims);
Test.GT = reshape(Test.PT*(Test.T'*img_gt(:)), Test.imdims);

Test.ESTimg = [];

%fprintf(sprintf('%d images are loaded for testing.\n', Test.N));

end