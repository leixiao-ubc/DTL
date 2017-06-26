function [f] = test_evaluateInfluence( in, lut_offset, lut_step, lut_f)
% apply the influence functions (1D) 

size_in = size(in);
x = (in(:)'); % row vector    
[f] = lut_eval(x, lut_offset, lut_step, lut_f, 0, 0, 0);  
f = reshape(f, size_in);
   