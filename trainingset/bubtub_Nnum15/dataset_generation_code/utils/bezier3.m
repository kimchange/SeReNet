function [X,Y,Z]=bezier3(x,y,z)
n=length(x);
t=linspace(0,1,5000);
X=0;Y=0;Z=0;
for k=0:n-1
    tmp=nchoosek(n-1,k)*t.^k.*(1-t).^(n-1-k);
    X=X+tmp*x(k+1);
    Y=Y+tmp*y(k+1);
    Z=Z+tmp*z(k+1);
end
% h=plot3(xx,yy,zz,'r');
% if nargout==3
%     X=xx;Y=yy;Z=zz;
% end
% if nargout==1
%     X=h;

end