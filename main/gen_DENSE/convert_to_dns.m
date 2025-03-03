function convert_to_dns(mag, phax, phay, pixel_spacing, ke, roi_epi_contours, roi_endo_contours, slice_cat, template_file, out_file)

    %%% Loading template %%%
    template = load(template_file, '-mat');
    n_frames = size(mag, 3);


    %%% Load images %%%

    % Load 'img'
    template.img{1} = mag;
    template.img{2} = phax;
    template.img{3} = phay;

    % Load 'dns'
    template.dns(1) = struct('Name',         'Simulated',...
                             'UID',          dicomuid,...
                             'Type',         'xy',...
                             'MagIndex',     [1 1 NaN],...
                             'PhaIndex',     [2 3 NaN],...
                             'Number',       n_frames,...
                             'PixelSpacing', pixel_spacing,...
                             'Scale',        [1 1 NaN],...
                             'EncFreq',      [ke ke NaN],...
                             'SwapFlag',     false,...
                             'NegFlag',      [false,false,false]);

    % Load 'seq'
    empty = repmat({''}, [n_frames, 1]);
    seq = template.seq;
    for num = 1:3
        seq(num).Width       = size(mag,2);
        seq(num).Height      = size(mag,1);
        seq(num).Filename    = empty;
        seq(num).FileModDate = empty;
        seq(num).FileSize    = empty;
        seq(num).MediaStorageSOPInstanceUID = empty;
        seq(num).InstanceCreationTime = empty;

        seq(num).SOPInstanceUID = empty;
        seq(num).AcquisitionTime = empty;
        seq(num).ContentTime = empty;
        seq(num).TriggerTime = num2cell((1:n_frames).');
        seq(num).NominalInterval = num2cell(1000 * ones(n_frames,1));

        seq(num).InstanceNumber = num2cell((1:n_frames).');

        seq(num).LargestPixelValue = 2^12;
        seq(num).WindowCenter = 2048;
        seq(num).WindowWidth = 2048;

        seq(num).CardiacNumberOfImages = n_frames;
        seq(num).NumberInSequence = n_frames;

        seq(num).DENSEindex = num2cell((1:n_frames).');
        seq(num).PixelSpacing = pixel_spacing;

        imagecomments = empty;
        if (num == 1)
            seq(num).DENSEid = 'mag.overall'; 
            seq(num).DENSEdata = struct('Number',     n_frames,...
                                        'Partition',  [1 1],...
                                        'Scale',      [],...
                                        'EncFreq',    [],...
                                        'SwapFlag',   0,...
                                        'NegFlag',    [0 0 0]);
            for i = 1:n_frames
                fmt = ['DENSE overall mag - Rep:0/1 Slc:0/1 Par:0/1 Phs:%d/%d ',...
                        'RCswap:0 RCSflip:0/0/0'];
                imagecomments{i} = sprintf(fmt, i-1, n_frames);
            end
        elseif (num == 2)
            seq(num).DENSEid = 'pha.x';
            seq(num).DENSEdata = struct('Number',     n_frames,...
                                        'Partition',  [1 1],...
                                        'Scale',      1,...
                                        'EncFreq',    ke,...
                                        'SwapFlag',   0,...
                                        'NegFlag',    [0 0 0]);
            for i = 1:n_frames
                fmt = ['DENSE x-enc pha - Scale:1.000000 EncFreq:%0.2f Rep:0/1 ',...
                        'Slc:0/1 Par:0/1 Phs:%d/%d RCswap:0 RCSflip:0/0/0'];
                imagecomments{i} = sprintf(fmt, ke, i-1, n_frames);
            end
        elseif (num == 3)
            seq(num).DENSEid = 'pha.y';
            seq(num).DENSEdata = struct('Number',     n_frames,...
                                        'Partition',  [1 1],...
                                        'Scale',      1,...
                                        'EncFreq',    ke,...
                                        'SwapFlag',   0,...
                                        'NegFlag',    [0 0 0]);
            for i = 1:n_frames
                fmt = ['DENSE y-enc pha - Scale:1.000000 EncFreq:%0.2f Rep:0/1 ',...
                        'Slc:0/1 Par:0/1 Phs:%d/%d RCswap:0 RCSflip:0/0/0'];
                imagecomments{i} = sprintf(fmt, ke, i-1, n_frames);
            end
        end

        seq(num).ImageComments = imagecomments;
            
    end


    %%% Load ROI data %%%
    template_roi = template.roi(1);

    for fn = fieldnames(template_roi)'
        new_roi.(fn{1}) = template_roi.(fn{1});
    end

    new_roi.UID = dicomuid;
    new_roi.Name = slice_cat;
    new_roi.CorrectedNames = slice_cat;

    nbr_frames = size(roi_endo_contours, 2);
    new_roi.Position = cell(nbr_frames,2);
    new_roi.IsClosed = cell(nbr_frames,2);
    new_roi.IsCorner = cell(nbr_frames,2);
    new_roi.IsCurved = cell(nbr_frames,2);

    for i = 1:nbr_frames
        new_roi.Position{i,1} = roi_epi_contours{1,i};
        new_roi.Position{i,2} = roi_endo_contours{1,i};
        new_roi.IsClosed{i,1} = 1;
        new_roi.IsClosed{i,2} = 1;
        new_roi.IsCorner{i,1} = zeros(size(roi_epi_contours{1,i},1),1);
        new_roi.IsCorner{i,2} = zeros(size(roi_endo_contours{1,i},1),1);
        new_roi.IsCurved{i,1} = ones(size(roi_epi_contours{1,i},1),1);
        new_roi.IsCurved{i,2} = ones(size(roi_endo_contours{1,i},1),1);
    end
    % for i = nbr_frames+1:size(new_roi.Position,1)
    %     new_roi.Position{i,1} = [];
    %     new_roi.Position{i,2} = [];
    %     new_roi.IsClosed{i,1} = [];
    %     new_roi.IsClosed{i,2} = [];
    %     new_roi.IsCorner{i,1} = [];
    %     new_roi.IsCorner{i,2} = [];
    %     new_roi.IsCurved{i,1} = [];
    %     new_roi.IsCurved{i,2} = [];
    % end

    template.roi(1) = new_roi;

    %%% Save final .dns workspace %%%
    roi = template.roi;
    dns = template.dns;
    img = template.img;
    save(out_file, 'dns', 'img', 'roi', 'seq');

end