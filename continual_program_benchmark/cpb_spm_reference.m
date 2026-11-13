function info = cpb_spm_reference(report, cfg)
%CPB_SPM_REFERENCE Independent SPM12 check after continual-learning outputs.
% Checks learned cue recognition at stages 1,3,6 of the first replication,
% for each of six cue symbols (18 cases). SPM receives only a learned cue
% likelihood, an availability prior and an observed cue. No task identity,
% true route, code posterior or prototype posterior is supplied to SPM.
% This static T=1 reference does not validate active query selection, motor
% execution, continual-learning claims, or a neural implementation of EFE.
    if nargin < 2, cfg = struct(); end
    assert(isstruct(report) && isfield(report, 'checkpoints') && ...
        isstruct(report.checkpoints) && ~isempty(report.checkpoints), ...
        'cpb:SPMReport', 'A completed report with nonempty checkpoints is required.');
    checkpoints = report.checkpoints;
    fields = {'replication', 'stage', 'cue_counts', 'q_active'};
    for k = 1:numel(fields)
        assert(isfield(checkpoints, fields{k}), 'cpb:SPMCheckpoint', ...
            'Checkpoint field %s is missing.', fields{k});
    end
    first_replication = min([checkpoints.replication]);
    stages = [1 3 6];
    selected = zeros(1, numel(stages));
    for k = 1:numel(stages)
        found = find([checkpoints.replication] == first_replication & [checkpoints.stage] == stages(k));
        assert(isscalar(found), 'cpb:SPMCheckpoint', ...
            'Exactly one checkpoint is required at stage %d of replication %d.', stages(k), first_replication);
        selected(k) = found;
    end
    if isfield(cfg, 'spm_root') && ~isempty(cfg.spm_root)
        spm_root = cfg.spm_root;
    else
        spm_root = getenv('SPM12_ROOT');
        if isempty(spm_root), spm_root = fullfile(getenv('HOME'), 'tools', 'spm12'); end
    end
    solveroriginalpath = fullfile(spm_root, 'toolbox', 'DEM', 'spm_MDP_VB_X.m');
    assert(exist(solveroriginalpath, 'file') == 2, 'cpb:SPMMissing', ...
        'Original SPM12 solver is required but absent: %s', solveroriginalpath);
    root_files = {'spm_softmax.m', 'spm_dot.m', 'spm_vec.m', 'spm_cross.m', ...
        'spm_zeros.m', 'spm_KL_dir.m', 'spm_betaln.m', 'spm_psi.m', ...
        'spm_unvec.m', 'spm_length.m'};
    dem_files = {'spm_MDP_VB_X.m', 'spm_MDP_check.m', 'spm_MDP_G.m'};
    old_path = path;
    old_rng = rng;
    cache = tempname;
    [ok, message] = mkdir(cache);
    assert(ok, 'cpb:SPMCache', 'Cannot create runtime cache: %s', message);
    guard = onCleanup(@() restore_environment(old_path, old_rng, cache));
    addpath(fullfile(fileparts(mfilename('fullpath')), 'vendor_mpb'));
    provenance = copy_source_bytes(spm_root, cache, root_files);
    provenance = [provenance, copy_source_bytes(fullfile(spm_root, 'toolbox', 'DEM'), cache, dem_files)];
    addpath(cache, '-begin');
    solverruntimepath = which('spm_MDP_VB_X');
    assert(strcmp(solverruntimepath, fullfile(cache, 'spm_MDP_VB_X.m')), ...
        'cpb:SPMResolution', 'The cached original solver must resolve first.');
    source_text = fileread(solverruntimepath);
    source_revision = regexp(source_text, '\$Id:[^\r\n]*', 'match', 'once');
    if isempty(source_revision), source_revision = 'No revision marker in installed source'; end
    sourcehashes = rmfield(provenance, {'originalpath', 'runtimepath', 'nbytes', 'byte_identical'});
    n = 6;
    p = mpb_parameters();
    percheck = struct([]);
    check_index = 0;
    for selected_index = selected
        checkpoint = checkpoints(selected_index);
        seen = 1:n <= checkpoint.stage;
        counts = checkpoint.cue_counts;
        assert(isequal(size(counts), [n n]) && all(isfinite(counts(:))) && all(counts(:) > 0), ...
            'cpb:SPMCueCounts', 'Learned cue counts must be positive and finite.');
        L = bsxfun(@rdivide, counts, sum(counts, 2))';
        prior = double(seen(:)) / sum(seen);
        for observed = 1:n
            check_index = check_index + 1;
            % Prototype output is computed separately; none enters the MDP.
            actual = cpb_recognize(counts, checkpoint.q_active, seen, observed, p);
            analytic = L(observed, :)' .* prior;
            analytic = analytic / sum(analytic);
            outcome = zeros(n, 1);
            outcome(observed) = 1;
            mdp = struct();
            mdp.A = {L};
            mdp.B = {eye(n)};
            mdp.D = {prior};
            mdp.O = {outcome};
            mdp.o = 0;
            mdp.tau = 1;
            mdp.eta = 0;
            solved = spm_MDP_VB_X(mdp, struct('plot', 0));
            posterior = full(solved.X{1}(:, 1));
            assert(solved.T == 1 && isempty(solved.u), 'cpb:SPMMode', ...
                'Reference must contain one observation and no actions or policies.');
            analytic_error = max(abs(posterior - analytic));
            assert(analytic_error < 1e-10, 'cpb:SPMAnalytic', ...
                'SPM differs from analytic Bayes in case %d by %.3g.', check_index, analytic_error);
            posterior_program_spm = posterior' * checkpoint.q_active;
            unseen_mass = sum(posterior(~seen));
            percheck(check_index) = struct('index', check_index, ...
                'replication', first_replication, 'stage', checkpoint.stage, ...
                'observation', observed, 'prior', prior', 'likelihood', L, ...
                'posterior_actual', actual.q_context, 'posterior_spm', posterior', ...
                'posterior_analytic', analytic', 'maxerror', max(abs(posterior' - actual.q_context)), ...
                'maxanalyticerror', analytic_error, 'unseen_mass_spm', unseen_mass, ...
                'unseen_mass_actual', sum(actual.q_context(~seen)), ...
                'program_mixture_error', max(abs(posterior_program_spm - actual.q_program))); %#ok<AGROW>
        end
    end
    info = struct('nchecks', numel(percheck), 'percheck', percheck, ...
        'maxerror', max([percheck.maxerror]), 'maxanalyticerror', max([percheck.maxanalyticerror]), ...
        'max_program_mixture_error', max([percheck.program_mixture_error]), ...
        'max_unseen_mass_spm', max([percheck.unseen_mass_spm]), ...
        'max_unseen_mass_actual', max([percheck.unseen_mass_actual]), ...
        'source_revision', source_revision, 'sourcehashes', sourcehashes, ...
        'source_provenance', provenance, 'dependency_files', {[root_files dem_files]}, ...
        'solveroriginalpath', solveroriginalpath, 'solverruntimepath', solverruntimepath, ...
        'mode', 'Static categorical recognition, T=1, no policies/actions; after benchmark outcomes', ...
        'scope', 'Learned cue likelihoods, first replication, stages 1/3/6, all six observed cues', ...
        'zero_prior_note', ['Prototype masks unavailable memories exactly. Installed SPM uses log(A+1e-16), ' ...
            'so zero prior entries can acquire finite precision probability; raw differences are reported.'], ...
        'posterior_source', 'X{1}(:,1), no future observations', 'runtime', version, ...
        'runtime_cache_removed', false, 'path_restored', false, 'rng_restored', false);
    clear guard
    info.runtime_cache_removed = ~exist(cache, 'dir');
    info.path_restored = strcmp(path, old_path);
    info.rng_restored = isequal(rng, old_rng);
    info.cleanupstats = struct('runtime_cache_removed', info.runtime_cache_removed, ...
        'path_restored', info.path_restored, 'rng_restored', info.rng_restored);
    assert(info.runtime_cache_removed && info.path_restored && info.rng_restored, ...
        'cpb:SPMCleanup', 'Runtime cache, path or random state was not restored.');
end

function provenance = copy_source_bytes(source, destination, files)
% Byte copies avoid cloud filesystem xattr operations; sources are unchanged.
    provenance = struct([]);
    for k = 1:numel(files)
        sourcefile = fullfile(source, files{k});
        fid = fopen(sourcefile, 'rb');
        assert(fid >= 0, 'cpb:SPMSourceRead', 'Cannot read original dependency: %s', sourcefile);
        guard = onCleanup(@() fclose(fid));
        bytes = fread(fid, Inf, '*uint8');
        clear guard
        runtimefile = fullfile(destination, files{k});
        fid = fopen(runtimefile, 'wb');
        assert(fid >= 0, 'cpb:SPMCacheWrite', 'Cannot write cached dependency: %s', files{k});
        guard = onCleanup(@() fclose(fid));
        count = fwrite(fid, bytes, 'uint8');
        assert(count == numel(bytes), 'cpb:SPMCacheWrite', 'Incomplete cached dependency: %s', files{k});
        clear guard
        fid = fopen(runtimefile, 'rb');
        assert(fid >= 0, 'cpb:SPMCacheRead', 'Cannot verify cached dependency: %s', files{k});
        guard = onCleanup(@() fclose(fid));
        copied = fread(fid, Inf, '*uint8');
        clear guard
        assert(isequal(bytes, copied), 'cpb:SPMByteIdentity', 'Cached dependency differs: %s', files{k});
        provenance(k) = struct('filename', files{k}, 'originalpath', sourcefile, ...
            'runtimepath', runtimefile, 'nbytes', numel(bytes), 'byte_identical', true, ...
            'sha256', sha256_bytes(bytes)); %#ok<AGROW>
    end
end

function digest = sha256_bytes(bytes)
    if exist('OCTAVE_VERSION', 'builtin')
        digest = hash('sha256', char(bytes(:)'));
    elseif usejava('jvm')
        md = java.security.MessageDigest.getInstance('SHA-256');
        md.update(typecast(bytes(:), 'int8'));
        digest = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
    else
        error('cpb:SPMHash', 'SHA-256 provenance requires Octave hash or the MATLAB JVM.');
    end
end

function restore_environment(old_path, old_rng, cache)
    path(old_path);
    rng(old_rng);
    if exist(cache, 'dir'), rmdir(cache, 's'); end
end
