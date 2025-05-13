function imgstack = imread3d(name)
%imread3d reads 3D TIFF stack
%   Inputs:
%       name - name with path of TIFF stack

    tifinfo = imfinfo(name);

    Width = tifinfo(1).Width;
    Height = tifinfo(1).Height;
    Depth = numel(tifinfo);
    imgstack = zeros(Height, Width, Depth);  % double by default
    for i = 1:Depth
        imgstack(:,:,i) = imread(name, i);
    end
    
end

