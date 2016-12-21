function [d, dy, dh] = checkgrad_custom_lbfgs(f, X, e, Training, Model, Data)%, P1, P2, P3, P4, P5);

global check_range
% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad('f', X, e, P1, P2, ...)
%
% where X is the argument and e is the small perturbation used for the finite
% differences. and the P1, P2, ... are optional additional parameters which
% get passed to f. The function f should be of the type 
%
% [fX, dfX] = f(X, P1, P2, ...)
%
% where fX is the function value and dfX is a vector of partial derivatives.
%
% Carl Edward Rasmussen, 2001-08-01.

argstr = [f, '(X'];                            % assemble function call strings
argstrd = [f, '(X+dx'];
%for i = 1:(nargin - 3)
%  argstr = [argstr, ',P', int2str(i)];
%  argstrd = [argstrd, ',P', int2str(i)];
%end
argstr = [argstr, ',Training', ',Model', ',Data'];
argstrd = [argstrd,',Training', ',Model', ',Data'];

argstr = [argstr, ')'];
argstrd = [argstrd, ')'];

[y dy] = eval(argstr);                         % get the partial derivatives dy

dh = zeros(length(X),1) ;

target_cof = 1:length(X); 

for i = 1:length(target_cof)
  j = target_cof(i);
  fprintf('_%.3f_', i/length(target_cof));
  dx = zeros(length(X),1);
  dx(j) = dx(j) + e;                               % perturb a single dimension
  y2 = eval(argstrd);
  dx = -dx ;
  y1 = eval(argstrd);
  dh(j) = (y2 - y1)/(2*e);
end

format shortE

fprintf('\nanalytic | numeric');
disp([dy(target_cof) dh(target_cof)])                                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
