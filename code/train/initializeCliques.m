function [] = initializeCliques()

global Training Model

% code borrowed from CSF paper
% index image with circular padding
fw = Model.filterWidth;
fdims = [fw, fw]; 
npixels = prod(Training.imdims);
idx = reshape(1:npixels,Training.imdims);

circpadidx = padarray(idx, (fdims-1)/2, 'circular','both');
Training.cliques_of_circ_conv = flipud(im2col(circpadidx,fdims,'sliding'));


end