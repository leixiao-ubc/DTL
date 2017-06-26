function y = edgewin(x,sw,windowfun)
% EDGEWIN edgetapers an input image with an window function of
% specified size sw.
%
% Input: x    image
%        sw   size of fadeout
%        windowfun   optional window function (@barthannwin per default)
% 
% Michael Hirsch (c) 2010    
    
if ~exist('windowfun','var')||isempty(windowfun), windowfun = @barthannwin; end
    
% Make sure that sw is odd sized
swd = 1-mod(sw,2);
if any(swd), sw = sw + swd; end
sw2 = floor(sw/2);
window = win_array2(windowfun, sw);

sx  = size(x);
swd = sx -sw;

w = cell(3,3);
w{1,1} = window(1:sw2(1),1:sw2(2));
w{1,3} = window(1:sw2(1),end-sw2(2)+1:end);
w{3,1} = window(end-sw2(1)+1:end,1:sw2(2));
w{3,3} = window(end-sw2(1)+1:end,end-sw2(2)+1:end);
w{2,2} = ones(swd+1);
w{1,2} = repmat(window(1:sw2(1),sw2(2)+1),1,swd(2)+1);
w{3,2} = repmat(window(end-sw2(1)+1:end,sw2(2)+1),1,swd(2)+1);
w{2,1} = repmat(window(sw2(1)+1,1:sw2(2)),swd(1)+1,1);
w{2,3} = repmat(window(sw2(1)+1,end-sw2(2)+1:end),swd(1)+1,1);
    
y = x .* cell2mat(w);
    
    
    