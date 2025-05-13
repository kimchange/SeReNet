function output = normalize_percentile(inp, low, high)
%myFun - Description
%
% Syntax: output = myFun(input)
%
% Long description
    low_inp = prctile(inp(:), low);
    high_inp = prctile(inp(:), high);
    output = (inp - low_inp) ./ (high_inp - low_inp + 1e-9);

end