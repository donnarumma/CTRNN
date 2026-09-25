function info = mpb_spm_reference(report, cfg)
%MPB_SPM_REFERENCE Independent SPM12 check of six-context static inference.
% Call only after all prototype outcomes have been computed. SPM receives a
% likelihood, prior and observation, never the prototype posterior or route.
% This checks T=1 inference only, not policy selection, G, or replanning.
    if nargin < 2, cfg = struct(); end
    assert(isstruct(report) && isfield(report, 'belief_checks') && ...
        isfield(report, 'protocol'), 'mpb:SPMReport', ...
        'A completed benchmark report with belief_checks and protocol is required.');
    checks = report.belief_checks;
    assert(isstruct(checks) && ~isempty(checks), 'mpb:SPMChecks', ...
        'report.belief_checks must be a nonempty structure array.');
    required = {'regime', 'initial_cue', 'probe_cue', 'prior', 'posterior'};
    for j = 1:numel(required)
        assert(isfield(checks, required{j}), 'mpb:SPMCheckField', ...
            'Missing reference check field %s.', required{j});
    end
    protocol_fields = {'regime_names', 'initial_reliability', 'probe_reliability'};
    for j = 1:numel(protocol_fields)
        assert(isfield(report.protocol, protocol_fields{j}), 'mpb:SPMProtocolField', ...
            'Missing protocol field %s.', protocol_fields{j});
    end
    if isfield(cfg, 'spm_root') && ~isempty(cfg.spm_root)
        spm_root = cfg.spm_root;
    else
        spm_root = getenv('SPM12_ROOT');
        if isempty(spm_root), spm_root = fullfile(getenv('HOME'), 'tools', 'spm12'); end
    end
    solveroriginalpath = fullfile(spm_root, 'toolbox', 'DEM', 'spm_MDP_VB_X.m');
    assert(exist(solveroriginalpath, 'file') == 2, 'mpb:SPMMissing', ...
        'Original SPM12 solver is required but absent: %s', solveroriginalpath);
    root_files = {'spm_softmax.m', 'spm_dot.m', 'spm_vec.m', 'spm_cross.m', ...
        'spm_zeros.m', 'spm_KL_dir.m', 'spm_betaln.m', 'spm_psi.m', ...
        'spm_unvec.m', 'spm_length.m'};
    dem_files = {'spm_MDP_VB_X.m', 'spm_MDP_check.m', 'spm_MDP_G.m'};
    old_path = path;
    old_rng = rng;
    cache = tempname;
    [ok, message] = mkdir(cache);
    assert(ok, 'mpb:SPMCache', 'Cannot create runtime cache: %s', message);
    guard = onCleanup(@() restore_environment(old_path, old_rng, cache));
    provenance = copy_source_bytes(spm_root, cache, root_files);
    provenance = [provenance, copy_source_bytes(fullfile(spm_root, 'toolbox', 'DEM'), cache, dem_files)];
    addpath(cache, '-begin');
    solverruntimepath = which('spm_MDP_VB_X');
    assert(strcmp(solverruntimepath, fullfile(cache, 'spm_MDP_VB_X.m')), ...
        'mpb:SPMResolution', 'The cached original solver is not the resolved function.');
    source_text = fileread(solverruntimepath);
    source_revision = regexp(source_text, '\$Id:[^\r\n]*', 'match', 'once');
    if isempty(source_revision), source_revision = 'No revision marker in installed source'; end

    n = 6;
    percheck = struct([]);
    for k = 1:numel(checks)
        regime_index = find(strcmp(checks(k).regime, report.protocol.regime_names));
        assert(isscalar(regime_index), 'mpb:SPMRegime', ...
            'Check %d must match exactly one protocol regime.', k);
        initial_cue = checks(k).initial_cue;
        probe_cue = checks(k).probe_cue;
        assert(isnumeric(initial_cue) && isscalar(initial_cue) && any(initial_cue == 1:n), ...
            'mpb:SPMOutcome', 'Check %d has an invalid initial cue.', k);
        assert(isnumeric(probe_cue) && isscalar(probe_cue) && any(probe_cue == 0:n), ...
            'mpb:SPMOutcome', 'Check %d has an invalid probe cue.', k);
        if probe_cue == 0
            reliability = report.protocol.initial_reliability(regime_index);
            observed = initial_cue;
            update_kind = 'initial_cue';
        else
            reliability = report.protocol.probe_reliability;
            observed = probe_cue;
            update_kind = 'probe_cue';
        end
        assert(isnumeric(reliability) && isscalar(reliability) && ...
            isfinite(reliability) && reliability > 0 && reliability < 1, ...
            'mpb:SPMReliability', 'Check %d needs a reliability strictly between zero and one.', k);
        L = reliability * eye(n) + (1 - reliability) / (n - 1) * (ones(n) - eye(n));
        prior = checks(k).prior;
        actual = checks(k).posterior;
        assert(isnumeric(prior) && isreal(prior) && isequal(size(prior), [1, n]) && ...
            all(isfinite(prior)) && all(prior > 0) && abs(sum(prior) - 1) < 1e-12, ...
            'mpb:SPMPrior', 'Check %d needs a positive, normalized six-context row prior.', k);
        assert(isnumeric(actual) && isreal(actual) && isequal(size(actual), [1, n]) && ...
            all(isfinite(actual)) && all(actual >= 0) && abs(sum(actual) - 1) < 1e-12, ...
            'mpb:SPMActual', 'Check %d needs a normalized six-context prototype posterior.', k);
        prior = double(prior(:));
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
        assert(solved.T == 1 && isempty(solved.u), 'mpb:SPMMode', ...
            'SPM reference must have one observation and no action selection.');
        analytic_error = max(abs(posterior - analytic));
        assert(analytic_error < 1e-10, 'mpb:SPMAnalytic', ...
            'SPM differs from analytical Bayes in check %d by %.3g.', k, analytic_error);
        percheck(k) = struct('index', k, 'regime', checks(k).regime, ...
            'initial_cue', initial_cue, 'probe_cue', probe_cue, 'update_kind', update_kind, ...
            'observationindex', observed, 'likelihood', L, 'prior', prior', ...
            'posterioractual', actual, 'posterior_spm', posterior', ...
            'posterior_analytic', analytic', 'maxerror', max(abs(posterior' - actual)), ...
            'maxanalyticerror', analytic_error, 'solveroriginalpath', solveroriginalpath); %#ok<AGROW>
    end
    info = struct('percheck', percheck, 'maxerror', max([percheck.maxerror]), ...
        'maxanalyticerror', max([percheck.maxanalyticerror]), 'nchecks', numel(checks), ...
        'n_contexts', n, 'solveroriginalpath', solveroriginalpath, ...
        'solverruntimepath', solverruntimepath, 'source_revision', source_revision, ...
        'dependency_files', {[root_files, dem_files]}, 'source_provenance', provenance, ...
        'mode', 'Static HMM, T=1, no policies or actions; independent reference after prototype outputs', ...
        'update_time_constant', 1, 'posterior_source', 'X{1}(:,1), no future observations', ...
        'runtime', version, 'runtime_cache_removed', false, 'path_restored', false, 'rng_restored', false);
    clear guard
    info.runtime_cache_removed = ~exist(cache, 'dir');
    info.path_restored = strcmp(path, old_path);
    info.rng_restored = isequal(rng, old_rng);
    assert(info.runtime_cache_removed && info.path_restored && info.rng_restored, ...
        'mpb:SPMCleanup', 'SPM runtime cache, path, or random state was not restored.');
end

function provenance = copy_source_bytes(source, destination, files)
% Binary copy and read-back verification avoid metadata/xattr cloud operations.
    provenance = struct([]);
    for k = 1:numel(files)
        sourcefile = fullfile(source, files{k});
        fid = fopen(sourcefile, 'rb');
        assert(fid >= 0, 'mpb:SPMSourceRead', 'Cannot read original SPM dependency: %s', sourcefile);
        guard = onCleanup(@() fclose(fid));
        bytes = fread(fid, Inf, '*uint8');
        clear guard
        runtimefile = fullfile(destination, files{k});
        fid = fopen(runtimefile, 'wb');
        assert(fid >= 0, 'mpb:SPMCacheWrite', 'Cannot write cached dependency: %s', files{k});
        guard = onCleanup(@() fclose(fid));
        count = fwrite(fid, bytes, 'uint8');
        assert(count == numel(bytes), 'mpb:SPMCacheWrite', 'Incomplete cached dependency: %s', files{k});
        clear guard
        fid = fopen(runtimefile, 'rb');
        assert(fid >= 0, 'mpb:SPMCacheRead', 'Cannot verify cached dependency: %s', files{k});
        guard = onCleanup(@() fclose(fid));
        copied = fread(fid, Inf, '*uint8');
        clear guard
        assert(isequal(bytes, copied), 'mpb:SPMByteIdentity', 'Cached dependency differs: %s', files{k});
        provenance(k) = struct('filename', files{k}, 'originalpath', sourcefile, ...
            'runtimepath', runtimefile, 'nbytes', numel(bytes), 'byte_identical', true); %#ok<AGROW>
    end
end

function restore_environment(old_path, old_rng, cache)
    path(old_path);
    rng(old_rng);
    if exist(cache, 'dir'), rmdir(cache, 's'); end
end
