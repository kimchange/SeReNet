function SE = getbubbleSE(rxy,rz)
%GETBUBBLESE 
nhood = zeros(2*rxy+1,2*rxy+1,2*rz+1);
[ii,jj,kk] = ndgrid(-rxy:rxy,-rxy:rxy,-rz:rz);
dpoint = ii.^2 / rxy^2  + jj.^2 / rxy^2 + kk.^2 / rz^2;
% nhood((dpoint <= 1) & (dpoint > ((rz - 1)/rz) ^2)) = 1;
nhood((dpoint <= 1.21) & (dpoint >= 0.81)) = 1;

SE = strel('arbitrary',nhood);
% 
end

