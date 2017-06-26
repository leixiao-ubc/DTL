function [out] = runBM3D(in, sigma, dnv)

[~, out] = BM3D(1, in, sigma, 'np', 0); % no reference image
fprintf('  (bm3d sigma %.2f)  ', sigma);
out = dnv.*out;

end