function info = tva_spm_reference(checks, cfg)
%TVA_SPM_REFERENCE Independent SPM12 reference for static three-context Bayes.
% info = tva_spm_reference(checks, cfg)
% Each checks(k) contains:
%   likelihood       3x3, outcome rows / context columns; columns sum to one
%   prior            3x1 positive prior (normalized here if needed)
%   observationindex integer 1..3
%   posterioractual  1x3 posterior computed by the prototype
% Optional cfg.spm_root overrides SPM12_ROOT or ~/tools/spm12.
% Missing SPM is an error, not a silent skip. Original SPM source bytes are
% copied into a short-lived cache; no recursive path addition is performed.
% The prototype posterior is used ONLY for comparison after SPM has solved
% its own model. This reference does not drive the prototype or its actions.
if nargin < 2, cfg = struct(); end
assert(isstruct(checks) && ~isempty(checks), 'tva:SPMChecks', ...
    'checks must be a nonempty structure array.');
required = {'likelihood','prior','observationindex','posterioractual'};
for j = 1:numel(required)
    assert(isfield(checks, required{j}), 'tva:SPMCheckField', ...
        'Missing reference check field %s.', required{j});
end
if isfield(cfg, 'spm_root') && ~isempty(cfg.spm_root)
    spm_root = cfg.spm_root;
else
    spm_root = getenv('SPM12_ROOT');
    if isempty(spm_root), spm_root = fullfile(getenv('HOME'),'tools','spm12'); end
end
solveroriginalpath = fullfile(spm_root,'toolbox','DEM','spm_MDP_VB_X.m');
assert(exist(solveroriginalpath,'file') == 2, 'tva:SPMMissing', ...
    'Original SPM12 solver is required but absent: %s', solveroriginalpath);

root_files = {'spm_softmax.m','spm_dot.m','spm_vec.m','spm_cross.m', ...
    'spm_zeros.m','spm_KL_dir.m','spm_betaln.m','spm_psi.m','spm_unvec.m','spm_length.m'};
dem_files = {'spm_MDP_VB_X.m','spm_MDP_check.m','spm_MDP_G.m'};
old_path = path; old_rng = rng;
cache = tempname;
[ok,message] = mkdir(cache);
assert(ok,'tva:SPMCache','Cannot create runtime cache: %s',message);
guard = onCleanup(@() restore_environment(old_path,old_rng,cache));
copy_source_bytes(spm_root,cache,root_files);
copy_source_bytes(fullfile(spm_root,'toolbox','DEM'),cache,dem_files);
addpath(cache,'-begin');
solverruntimepath = which('spm_MDP_VB_X');
assert(strcmp(solverruntimepath,fullfile(cache,'spm_MDP_VB_X.m')), ...
    'tva:SPMResolution','The cached original SPM solver is not the resolved function.');

percheck = struct([]);
for k = 1:numel(checks)
    L = checks(k).likelihood; prior = checks(k).prior;
    observed = checks(k).observationindex; actual = checks(k).posterioractual;
    assert(isnumeric(L) && isreal(L) && isequal(size(L),[3 3]) && ...
        all(isfinite(L(:))) && all(L(:)>=0) && max(abs(sum(L,1)-1))<1e-12, ...
        'tva:SPMLikelihood','Check %d must have a nonnegative column-stochastic 3x3 likelihood.',k);
    assert(isnumeric(prior) && isreal(prior) && isequal(size(prior),[3 1]) && ...
        all(isfinite(prior)) && all(prior>0), ...
        'tva:SPMPrior','Check %d must have a finite positive 3x1 prior.',k);
    assert(isnumeric(observed) && isscalar(observed) && any(observed==[1 2 3]), ...
        'tva:SPMOutcome','Check %d must have observationindex 1, 2, or 3.',k);
    assert(isnumeric(actual) && isreal(actual) && isequal(size(actual),[1 3]) && ...
        all(isfinite(actual)) && all(actual>=0) && abs(sum(actual)-1)<1e-12, ...
        'tva:SPMActual','Check %d must supply a normalized 1x3 prototype posterior.',k);
    prior = double(prior)/sum(prior); L = double(L);
    analytic = L(observed,:)'.*prior;
    evidence = sum(analytic);
    assert(evidence>0,'tva:SPMImpossibleObservation','Check %d has zero observation evidence.',k);
    analytic = analytic/evidence;
    outcome = zeros(3,1); outcome(observed)=1;
    mdp = struct();
    mdp.A = {L}; mdp.B = {eye(3)}; mdp.D = {prior}; mdp.O = {outcome};
    mdp.o = 0; mdp.tau = 1; mdp.eta = 0;
    solved = spm_MDP_VB_X(mdp,struct('plot',0));
    posterior = full(solved.X{1}(:,1));
    assert(solved.T==1 && isempty(solved.u),'tva:SPMMode', ...
        'SPM reference must have one observation and no action selection.');
    analytic_error = max(abs(posterior-analytic));
    assert(analytic_error<1e-10,'tva:SPMAnalytic', ...
        'SPM differs from analytical Bayes in check %d by %.3g.',k,analytic_error);
    percheck(k) = struct('index',k,'observationindex',observed, ...
        'likelihood',L,'prior',prior,'posterioractual',actual, ...
        'posterior_spm',posterior','posterior_analytic',analytic', ...
        'maxerror',max(abs(posterior'-actual)),'maxanalyticerror',analytic_error); %#ok<AGROW>
end
info = struct('percheck',percheck,'maxerror',max([percheck.maxerror]), ...
    'maxanalyticerror',max([percheck.maxanalyticerror]), ...
    'nchecks',numel(checks),'solveroriginalpath',solveroriginalpath, ...
    'solverruntimepath',solverruntimepath,'dependency_files',{[root_files dem_files]}, ...
    'mode','Static HMM, T=1, no policies or actions; independent reference only', ...
    'update_time_constant',1,'posterior_source','X{1}(:,1), no future observations', ...
    'runtime',version,'runtime_cache_removed',false);
clear guard
info.runtime_cache_removed = ~exist(cache,'dir');
assert(info.runtime_cache_removed,'tva:SPMCacheCleanup','SPM runtime cache was not removed.');
end

function copy_source_bytes(source,destination,files)
% Binary copying avoids slow cloud metadata/xattr operations and preserves
% the installed original MATLAB source bytes exactly.
for k = 1:numel(files)
    sourcefile = fullfile(source,files{k});
    fid = fopen(sourcefile,'rb');
    assert(fid>=0,'tva:SPMSourceRead','Cannot read original SPM dependency: %s',sourcefile);
    guard = onCleanup(@() fclose(fid));
    bytes = fread(fid,Inf,'*uint8'); clear guard
    fid = fopen(fullfile(destination,files{k}),'wb');
    assert(fid>=0,'tva:SPMCacheWrite','Cannot write cached SPM dependency: %s',files{k});
    guard = onCleanup(@() fclose(fid));
    count = fwrite(fid,bytes,'uint8');
    assert(count==numel(bytes),'tva:SPMCacheWrite','Incomplete cached dependency: %s',files{k});
    clear guard
end
end

function restore_environment(old_path,old_rng,cache)
path(old_path); rng(old_rng);
if exist(cache,'dir'), rmdir(cache,'s'); end
end
