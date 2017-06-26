function I = fix_bndry(I,k,alpha, PT, imdims)


I = reshape(PT * I(:), imdims);
fft_k = psf2otf(k, imdims);

if(size(k,1)>1)
    ntapers = 4;
    for i = 1:ntapers
       blurredI = real(ifft2(fft2(I).*fft_k)); % important to be circular
       I = alpha.*I + (1-alpha).*blurredI;
    end
end
