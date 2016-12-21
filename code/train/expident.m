function [result] = expident(xraw)
% expident(xraw) = xraw (if xraw>=1), or exp(xraw-1) (else).

size_x = size(xraw);
result.value = zeros(size_x);
result.grad = zeros(size_x);

result.value = exp(xraw-1);
result.grad = result.value;

mask = find(xraw>=1);
result.value(mask) = xraw(mask);
result.grad(mask) = 1;

end