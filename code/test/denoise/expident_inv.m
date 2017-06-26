function [xraw] = expident_inv(x)
% expident(xraw) = xraw (if xraw>=1), or exp(xraw-1) (else).

assert(x>0);
assert(length(x)==1);

if(x>=1)
    xraw = x;
else
    xraw = 1 + log(x);
end

end