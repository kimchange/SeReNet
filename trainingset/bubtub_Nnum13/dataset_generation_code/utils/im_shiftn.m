function new_im = im_shiftn(img, SHIFT)
%% Input:
% @img: the nD image  
% @SHIFT: the shift in the n dimensions, length(SHIFT) == length(size(img))
%% Output:
% new_im: the shifted nD image  
%
%
%    Contact: kimchange (DO NOT REALLY CONTACT)
%    Date  : 18/7/2021

new_im = padarray(img, abs(SHIFT),0,'both');
command = 'new_im(';
for dim = 1:length(SHIFT)
    begin_ = 1 - SHIFT(dim) + abs(SHIFT(dim));
    end_ = begin_ + size(img, dim) -1;
    command = [command, num2str(begin_), ':', num2str(end_), ','];
end
command(end) = ')';
new_im = eval(command);

end