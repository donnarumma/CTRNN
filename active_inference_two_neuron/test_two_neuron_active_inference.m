function report = test_two_neuron_active_inference(cfg)
% Behavioral and numerical regression checks for the default experiment.
% Uses the actual CTRNN constructors/integrator and unmodified SPM MDP solver.
if nargin < 1, cfg = struct(); end
cfg.make_plots = false;
cfg.save_results = false;
cfg.trials_per_context = 16;
cfg.seed = 20260922;
previous_path = path;
paths = ai2_setup(cfg);
guard = onCleanup(@() cleanup_test(paths,previous_path)); %#ok<NASGU>
backend_cfg = cfg;
backend_cfg.ctrnn_root = paths.runtime_ctrnn;
backend = ai2_test_backend(backend_cfg); %#ok<NASGU>
clear guard
report = run_two_neuron_active_inference(cfg);
report.backend_validation = backend;
active = report.summary(strcmp({report.summary.condition},'active'));
passive = report.summary(strcmp({report.summary.condition},'passive'));
oracle = report.summary(strcmp({report.summary.condition},'oracle'));
assert(active.informative_probe_rate > .9,'ai2:NoExploration','Active agent does not seek information.');
assert(passive.informative_probe_rate == 0,'ai2:Passive','Passive baseline used forbidden probe.');
assert(oracle.informative_probe_rate < .1,'ai2:Oracle','Known context still triggers costly probing.');
assert(active.accuracy >= .8 && oracle.accuracy == 1,'ai2:Accuracy','Program selection failed.');
assert(passive.accuracy <= .7,'ai2:PassiveAccuracy','Unexpected passive accuracy: check information leakage.');
assert(active.mean_posterior_true > passive.mean_posterior_true+.2, ...
    'ai2:Beliefs','Informative action did not improve online beliefs.');
assert(all(diag(report.execution_rmse) < .04),'ai2:Emulation','Correct-code execution too inaccurate.');
assert(report.execution_rmse(1,2) > .12 && report.execution_rmse(2,1) > .12, ...
    'ai2:WrongCode','Wrong codes do not meaningfully change execution.');
for name = {'active','passive'}
    episodes = report.runs(strcmp({report.runs.condition},name{1}));
    policy = episodes(1).policies_initial;
    for episode = episodes
        assert(max(abs(episode.prior_online - [.5;.5])) < 1e-8,'ai2:Prior','Hidden-context leak.');
        assert(max(abs(episode.policies_initial-policy)) < 1e-8, ...
            'ai2:PolicyLeak','Initial policy depends on hidden context or future observations.');
        assert(abs(sum(episode.posterior_after_probe)-1) < 1e-8, ...
            'ai2:Normalization','Invalid posterior.');
    end
end

report.validation_passed = true;
fprintf('PASS: neural fidelity, fixed weights, information seeking, online beliefs, and controls.\n');
end

function cleanup_test(paths,previous_path)
path(previous_path);
if ~isempty(paths.cache_root) && exist(paths.cache_root,'dir'), rmdir(paths.cache_root,'s'); end
end
