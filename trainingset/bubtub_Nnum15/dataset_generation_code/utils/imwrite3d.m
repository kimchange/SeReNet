function imwrite3d(img, name, bitdepth)
%imwrite3d writes given 3D img in bitdepth 8 or 16 to given name with path
%   Inputs:
%       img - 3D image stack
%       name - file name string
%       bitdepth - 8 or 16

if bitdepth == 8
    assert(max(img(:)) <= 255, 'maximum of image data to be saved exceeds 8-bit 255');
    img = uint8(img);
elseif bitdepth == 16 
    assert(max(img(:)) <= 65535, 'image data to be saved exceeds 16-bit 65535');
    img = uint16(img);   
else
    error(['The bitdepth ' num2str(bitdepth) ' is not supported.']);
end
    

% imwrite(img(:,:,1), name);
imwrite(img(:,:,1), name, 'Compression', 'deflate');

for i = 2:size(img, 3)
    % imwrite((img(:,:,i)), name,  'WriteMode', 'append');
    imwrite((img(:,:,i)), name,  'WriteMode', 'append', 'Compression', 'deflate');
end

