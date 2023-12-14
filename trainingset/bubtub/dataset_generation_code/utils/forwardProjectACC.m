function TOTALprojection = forwardProjectACC( H, realspace)

Nnum = size(H,3);
zerospace = zeros(  size(realspace,1),   size(realspace,2), 'single');
TOTALprojection = zerospace;




for aa=1:Nnum,
    for bb=1:Nnum,
        for cc=1:size(realspace,3),

            
            Hs = squeeze(H( :, : ,aa,bb,cc));          
            tempspace = zerospace;
            tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc);
            % projection = conv2(tempspace, Hs, 'same');
            projection = conv2fft(tempspace, Hs);
            TOTALprojection = TOTALprojection + projection;            
        end
    end
end

