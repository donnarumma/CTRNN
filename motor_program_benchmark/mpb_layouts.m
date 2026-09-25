function layouts = mpb_layouts(c,regenerate)
%MPB_LAYOUTS Deterministic manifests with distinct start and four landmarks.
if nargin<2, regenerate=false; end
manifest=fullfile(fileparts(mfilename('fullpath')),'layout_manifest.mat');
if ~regenerate && exist(manifest,'file')==2
    saved=load(manifest,'layouts','manifest_protocol');
    seed_fields={'development_seed','evaluation_seed','transfer_seed','grid_size','transfer_grid_size'};
    for j=1:numel(seed_fields)
        name=seed_fields{j}; assert(isequal(c.(name),saved.manifest_protocol.(name)), ...
            'mpb:Manifest','A changed layout protocol requires a newly versioned manifest.');
    end
    names={'development','evaluation','geometry_transfer'};
    counts=[c.n_development,c.n_evaluation,c.n_transfer]; layouts=saved.layouts([]);
    for j=1:3
        eligible=find(strcmp({saved.layouts.split},names{j}));
        assert(counts(j)<=numel(eligible),'mpb:Manifest','Requested more layouts than the frozen manifest contains.');
        layouts=[layouts,saved.layouts(eligible(1:counts(j)))]; %#ok<AGROW>
    end
    return
end
previous=rng(); restore=onCleanup(@() rng(previous)); %#ok<NASGU>
counts=[c.n_development c.n_evaluation c.n_transfer];
seeds=[c.development_seed c.evaluation_seed c.transfer_seed];
sizes=[c.grid_size c.grid_size c.transfer_grid_size];
splits={'development','evaluation','geometry_transfer'};
layouts=struct('id',{},'split',{},'seed',{},'grid_size',{},'start',{},'landmarks',{});
for split=1:3
    rng(seeds(split),'twister');
    for k=1:counts(split)
        cells=randperm(sizes(split)^2,5);
        [rows,cols]=ind2sub([sizes(split) sizes(split)],cells);
        points=[rows(:),cols(:)];
        a=struct('id',sprintf('%s_%03d',splits{split},k),'split',splits{split}, ...
            'seed',seeds(split),'grid_size',sizes(split),'start',points(1,:), ...
            'landmarks',points(2:5,:));
        layouts(end+1)=a; %#ok<AGROW>
    end
end
end
