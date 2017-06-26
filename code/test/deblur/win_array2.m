function www = win_array2(wfun, sw);
www = wfun(sw(1)) * wfun(sw(2))';
www = www./max(www(:));
www = max(eps, www);
return