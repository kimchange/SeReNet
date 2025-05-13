% by kimchange 2022
% global shift correct
function [res,shift_1,shift_2,shift_3] = shift3_registration(inp, ref)
    % inp_mip = gpuArray(single(max(inp,[],3)));
    % ref_mip = gpuArray(single(max(ref,[],3)));
    inp_mip = max(inp,[],3);
    ref_mip = max(ref,[],3);
    % inp_mip(inp_mip>max(ref_mip,[],'all')) = max(ref_mip,[],'all');
    corr_map=gather(normxcorr2(ref_mip,inp_mip));
    [testa,testb]=find(corr_map==max(corr_map(:)));
    shift_1 = size(inp_mip,1) - testa;
    shift_2 = size(inp_mip,2) - testb;


    res_1 = im_shiftn(inp, [shift_1, shift_2, 0]);
    corr_z = zeros(1,size(res_1,3)+size(ref,3)-1);
    for ii = 1:length(corr_z)
        corr_z(1,ii) = sum(squeeze(mean(mean(   res_1(:,:,max(end-ii+1,1):min(end,end+size(ref,3)-ii)) .*   ref(:,:,max(1,ii-size(res_1,3)+1):min(end, ii) )    ,1),2)) );
    end
    testc = find(corr_z==max(corr_z(:)));
    shift_3 = -(size(res_1,3) - testc);
    res = im_shiftn(res_1, [0, 0, shift_3]);

end