function inds = getbeadinds(rxy,rz,h,w,d)
    nhood = zeros(2*rxy+1,2*rxy+1,2*rz+1);
    [ii,jj,kk] = ndgrid(-rxy:rxy,-rxy:rxy,-rz:rz);
    dpoint = ii.^2 / rxy^2  + jj.^2 / rxy^2 + kk.^2 / rz^2;
    % nhood((dpoint <= 1) & (dpoint > ((rz - 1)/rz) ^2)) = 1;
    nhood(dpoint <= 1.) = 1;
    volume = zeros(h,w,d);
    volume(1:size(nhood,1), 1:size(nhood,2), 1:size(nhood,3)) = nhood;
    inds = find(volume == 1) - sub2ind(size(volume), rxy+1, rxy+1, rz+1);
    end
    
    