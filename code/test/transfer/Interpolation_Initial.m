% simulate random projection
% c=0 denotes preserved pixels
function im_r=Interpolation_Initial(im_n,c)


% Delaunay triangulation based interpolation
[x,y]=find(c==0);
[M,N]=size(im_n);
[x1,y1]=meshgrid(1:M,1:N);
im_r=griddata(x,y,im_n(find(c==0)),x1,y1);
im_r=im_r';

% how to deal with NaN values
find(isnan(im_r))