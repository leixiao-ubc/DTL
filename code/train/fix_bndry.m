function I = fix_bndry(I,k,alpha)

global Training

I = reshape(Training.PT * I(:), Training.imdims);
   
if(size(k,1)>1)
    ntapers = 5;
    for i = 1:ntapers
       blurredI = imfilter(I, k, 'conv', 'circular'); % important to be circular
       I = alpha.*I + (1-alpha).*blurredI;
    end
end
