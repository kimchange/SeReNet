function fnames = get_all_abs_path(path,pattern)
    % By kimchange 20221221
    % given a folder, output all abs path of any document whose name has a pattern "string" in it
    arguments
        path (1,:) char = './'
        pattern (:,:) char = ''
    end
    fnames = {};
    subpath = dir(path);
    for ii = 1:length(subpath)
        if( isequal( subpath( ii ).name, '.' )||...
            isequal( subpath( ii ).name, '..'))
                continue;
        end
        if ( subpath( ii ).isdir )
            fnames = [fnames, get_all_abs_path(fullfile(subpath( ii ).folder,subpath( ii ).name), pattern)];
        else
            fname = {fullfile(subpath( ii ).folder,subpath( ii ).name)};
            if contains(fname,pattern)
                fnames = [fnames, fname  ];
            end
        end
    end
end