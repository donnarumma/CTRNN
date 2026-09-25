function paths = bci_setup(cfg)
% Use installed sources; a byte-for-byte temporary cache avoids cloud-mount scans.
% Callers can provide cfg.ctrnn_root/cfg.spm_root or CTRNN_ROOT/SPM12_ROOT.
if nargin < 1, cfg = struct(); end
paths.ctrnn_root = resolve_root(cfg, 'ctrnn_root', 'CTRNN_ROOT', 'CTRNN');
paths.spm_root = resolve_root(cfg, 'spm_root', 'SPM12_ROOT', 'spm12');
assert(exist(fullfile(paths.ctrnn_root, 'mulation.m'), 'file') == 2, ...
    'bci:CTRNNMissing', 'CTRNN root does not contain mulation.m: %s', paths.ctrnn_root);
assert(exist(fullfile(paths.spm_root, 'toolbox', 'DEM', 'spm_MDP_VB_X.m'), 'file') == 2, ...
    'bci:SPMMissing', 'SPM12 MDP solver not found at %s', paths.spm_root);
if ~isfield(cfg,'use_runtime_cache'), cfg.use_runtime_cache = true; end
paths.cache_root = '';
paths.runtime_ctrnn = paths.ctrnn_root;
paths.runtime_spm = paths.spm_root;
if cfg.use_runtime_cache
    paths.cache_root = tempname;
    paths.runtime_ctrnn = fullfile(paths.cache_root,'ctrnn');
    paths.runtime_spm = fullfile(paths.cache_root,'spm');
    mkdir(paths.runtime_ctrnn); mkdir(paths.runtime_spm);
    ctrnn_files = {'createCTRNN.m','PAR_CTRNN.m','mulation.m', ...
        'weights2program.m','runCTRNN.m','runCTRNN_NoAsymWay.m','sigmoid.m','NET_Noise.m'};
    spm_files = {'spm_softmax.m','spm_dot.m','spm_vec.m','spm_cross.m', ...
        'spm_zeros.m','spm_KL_dir.m','spm_betaln.m','spm_psi.m','spm_unvec.m','spm_length.m'};
    mdp_files = {'spm_MDP_VB_X.m','spm_MDP_check.m','spm_MDP_G.m'};
    fprintf('Reading installed CTRNN/SPM sources into a temporary runtime cache...\n');
    copy_sources(paths.ctrnn_root,paths.runtime_ctrnn,ctrnn_files);
    copy_sources(paths.spm_root,paths.runtime_spm,spm_files);
    copy_sources(fullfile(paths.spm_root,'toolbox','DEM'),paths.runtime_spm,mdp_files);
    paths.cached_ctrnn_files = ctrnn_files;
    paths.cached_spm_files = [spm_files mdp_files];
    addpath(paths.runtime_ctrnn); addpath(paths.runtime_spm);
else
    addpath(paths.ctrnn_root);
    addpath(paths.spm_root);
    addpath(fullfile(paths.spm_root,'toolbox','DEM'));
end
paths.solver = which('spm_MDP_VB_X');
paths.interpreter_builder = which('mulation');
paths.solver_source = fullfile(paths.spm_root,'toolbox','DEM','spm_MDP_VB_X.m');
paths.interpreter_builder_source = fullfile(paths.ctrnn_root,'mulation.m');
end

function copy_sources(source,destination,files)
% Binary I/O intentionally avoids copying cloud filesystem metadata/xattrs.
for k = 1:numel(files)
    fid = fopen(fullfile(source,files{k}),'rb');
    assert(fid >= 0,'bci:SourceRead','Cannot read source %s',files{k});
    guard = onCleanup(@() fclose(fid));
    bytes = fread(fid,Inf,'*uint8');
    clear guard
    fid = fopen(fullfile(destination,files{k}),'wb');
    assert(fid >= 0,'bci:CacheWrite','Cannot cache source %s',files{k});
    guard = onCleanup(@() fclose(fid));
    count = fwrite(fid,bytes,'uint8');
    assert(count == numel(bytes),'bci:CacheWrite','Incomplete cached file.');
    clear guard
end
end

function root = resolve_root(cfg, field, envname, dirname)
if isfield(cfg, field) && ~isempty(cfg.(field))
    root = cfg.(field);
else
    root = getenv(envname);
    if isempty(root)
        root = fullfile(getenv('HOME'), 'tools', dirname);
    end
end
end
