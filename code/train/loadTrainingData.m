function [] = loadTrainingData()

global Training Model Data

fprintf('reset random number generator.\n');
reset(RandStream.getGlobalStream);

%% load pre-created training data
load(Training.fn_trainingdata);
% the loaded .mat file should include  'data_GTimg', 'data_GTpsf',
% 'data_Meas', 'imdims', 'kdims', 'gtimdims'.
assert(Training.N<=size(data_Meas, 3));

fprintf('gtimdims = [%d, %d]', gtimdims(1), gtimdims(2));
fprintf('kdims = [%d, %d]', kdims(1), kdims(2));

%% pre-allocate memory and pre-compute some matrices
h = gtimdims(1); w = gtimdims(2); % before padding
wpsf = kdims(1);
Training.wpsf = wpsf;
Training.kdim = [wpsf, wpsf]; 
hi = h; wi = w;

%-------------------------------------------------------------------
filter_padding = Model.numDenoiseLayers*(Model.filterWidth-1)/2;
Training.wpad = max((Training.wpsf-1)/2, filter_padding); 
%-------------------------------------------------------------------

Training.imdims = [hi + Training.wpad*2, wi + Training.wpad*2];
Training.gtimdims = [hi, wi]; % don't forget to assign

% precompute truncating and padding matrix (code borrowed from CSF paper)
npixels = prod(Training.imdims);
t = Training.wpad; 
[r,c] = ndgrid(1+t:Training.imdims(1)-t, 1+t:Training.imdims(2)-t);
ind_int = sub2ind(Training.imdims, r(:), c(:));
d = zeros(Training.imdims); d(ind_int) = 1;
T = spdiags(d(:),0,npixels,npixels);
T = T(ind_int,:);
idximg = reshape(1:prod(Training.imdims-2*t),Training.imdims-2*t);
pad_idximg = padarray(idximg,[t t],'replicate','both');
P = sparse((1:npixels)',pad_idximg(:),ones(npixels,1),npixels,prod(Training.imdims-2*t));
PT = P*T; % first truncation, then padding
Training.T = T;
Training.PT = PT;

%% load data and simulate blurry measurements at original scale
wpsf = Training.wpsf; % blur kernel size at the original scale
Training.totalNumKernelPixels = Training.N*(wpsf^2);
Training.totalNumImagePixels = Training.N*prod(Training.imdims);   
for i = 1:Training.N
    Data{i}.valid_psf_width = psf_width(i+Training.img_startIDX);
    Data{i}.noise_sigma = noise_sigma(i+Training.img_startIDX);
    Data{i}.GTimg = data_GTimg(:, :, i+Training.img_startIDX).*Training.data_normalization_value;
    Data{i}.GTpsf  = data_GTpsf(:,:,i+Training.img_startIDX);     
    Data{i}.GTpsf_pad = zero_pad_psf(Data{i}.GTpsf, Training.imdims);
    
    if(Data{i}.valid_psf_width>1)
        fft_k = psf2otf(Data{i}.GTpsf_pad); % these can be pre-computed
        Data{i}.fft_k = fft_k;
        Data{i}.fft_kt = conj(fft_k);
        Data{i}.fft_ktk = abs(fft_k).^2;
    end
    
    mask = edgetaper_mask(Data{i}.GTpsf, Training.imdims);
    if(Training.use_quantized_input) % load 8-bit images directly
         blurred = data_Meas(:,:,i + Training.img_startIDX).*Training.data_normalization_value;
         Data{i}.Meas = fix_bndry(Training.T'*blurred(:), Data{i}.GTpsf, mask); 
    else
         blurred = imfilter(Data{i}.GTimg, Data{i}.GTpsf, 'conv', 'replicate');
         noisy = blurred + Training.data_normalization_value.*Data{i}.noise_sigma.*randn(size(blurred)); 
         Data{i}.Meas =  fix_bndry(Training.T'*(noisy(:)), Data{i}.GTpsf, mask); 
    end
end


%% save out blurry images
saveInput();

fprintf(sprintf('%d images are loaded for training.\n', Training.N));

end