function [xraw] = expident_inv(x)
% expident(xraw) = xraw (if xraw>=1), or exp(xraw-1) (else).

assert(isempty(find(x<=0, 1)));

xraw = x;
mask = find(x(:)<1);
xraw(mask) = 1 + log(x(mask));

end