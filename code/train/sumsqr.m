function [out] = sumsqr(in)

x = in.^2;
out = sum(x(:));

end