function report = run_two_neuron_active_inference(cfg)
% Active program identification using SPM12 and a fixed 14-neuron interpreter.
% report = run_two_neuron_active_inference(struct('make_plots',false));
% The two-neuron networks are the targets; this is a finite two-code example.
if nargin < 1, cfg = struct(); end
cfg = defaults(cfg);
previous_path = path;
previous_rng = rng;
paths = ai2_setup(cfg);
cleanup = onCleanup(@() restore_state(previous_path, previous_rng, paths)); %#ok<NASGU>
rng(cfg.seed, 'twister');
runtime_cfg = cfg;
runtime_cfg.ctrnn_root = paths.runtime_ctrnn;
bank = ai2_build_bank(runtime_cfg);
conditions = {'active','passive','oracle'};
options = struct('plot',0);

% Run each possible selected code on the informative validation input.
% Reuse one physical interpreter: the only varying inputs are the code values.
for context = 1:2
    target = ai2_simulate(bank.target_nets{context},bank.probes(2,:),[],bank.cfg);
    for selected = 1:2
        execution = ai2_simulate(bank.interpreter,bank.probes(2,:),bank.programs(selected,:),bank.cfg);
        delta = execution.readout - target.readout;
        execution_rmse(context,selected) = sqrt(mean(delta(:).^2)); %#ok<AGROW>
        if context == 1
            examples(selected).time = execution.time; %#ok<AGROW>
            examples(selected).output = execution.readout;
        end
    end
    examples(context).target = target.readout;
end

runs = struct([]);
index = 0;
for c = 1:numel(conditions)
    condition = conditions{c};
    fprintf('SPM condition: %s (%d trials per context)\n',condition,cfg.trials_per_context);
    for truth = 1:2
        for trial = 1:cfg.trials_per_context
            % Independent, reproducible episodes; no posterior carries between trials.
            rng(cfg.seed + 10000*c + 100*truth + trial,'twister');
            [mdp, design] = ai2_make_mdp(bank,truth,condition,cfg);
            solved = spm_MDP_VB_X(mdp,options);
            selected = solved.u(2,2)-2;
            assert(ismember(selected,[1 2]),'ai2:Action','Final action must execute a code.');
            index = index+1;
            runs(index).condition = condition;
            runs(index).truth = truth;
            runs(index).trial = trial;
            runs(index).probe = solved.u(2,1);
            runs(index).selected = selected;
            runs(index).correct = selected == truth;
            runs(index).sensor_outcome = solved.o(1,2);
            runs(index).prior_online = squeeze(solved.xn{1}(end,:,1,1))';
            runs(index).posterior_after_probe = squeeze(solved.xn{1}(end,:,2,2))';
            runs(index).execution_rmse = execution_rmse(truth,selected);
            runs(index).negative_efe_initial = solved.G(:,1);
            runs(index).policies_initial = solved.R(:,1);
            % SPM returns NEGATIVE expected free energy: larger is preferred.
            assert(max(abs(runs(index).prior_online - mdp.D{1})) < 1e-5, ...
                'ai2:PriorLeak','Initial posterior differs from intended prior.');
            for g = 1:numel(mdp.a)
                assert(isequal(solved.a{g},mdp.a{g}), ...
                    'ai2:Learning','Fixed model unexpectedly changed.');
            end
        end
    end
end

for c = 1:numel(conditions)
    chosen = runs(strcmp({runs.condition},conditions{c}));
    posterior = [chosen.posterior_after_probe];
    truth_idx = sub2ind(size(posterior),[chosen.truth],1:numel(chosen));
    summary(c).condition = conditions{c}; %#ok<AGROW>
    summary(c).n = numel(chosen);
    summary(c).accuracy = mean([chosen.correct]);
    summary(c).informative_probe_rate = mean([chosen.probe] == 2);
    summary(c).mean_posterior_true = mean(posterior(truth_idx));
    summary(c).mean_execution_rmse = mean([chosen.execution_rmse]);
    fprintf('  %-8s accuracy %.3f, informative %.3f, execution RMSE %.5f\n', ...
        summary(c).condition,summary(c).accuracy,summary(c).informative_probe_rate,summary(c).mean_execution_rmse);
end

report.config = cfg;
report.backend_config = bank.cfg;
report.paths = paths;
report.runtime = version;
report.programs = bank.programs;
report.probes = bank.probes;
report.target_predictions = bank.target_predictions;
report.interpreter_predictions = bank.interpreter_predictions;
report.model_high = design.model_high;
report.process_high = design.process_high;
report.execution_rmse = execution_rmse;
report.summary = summary;
report.runs = runs;
report.examples = examples;
report.interpreter_weights = bank.interpreter.internalWMatrix;
report.interpreter_external_weights = bank.interpreter.externalWMatrix;
report.interpreter_neurons = bank.interpreter.numUnits;
report.note = ['SPM supervisor over a continuous CTRNN interpreter; true process uses ' ...
    'compiled target responses, internal likelihood uses interpreter responses.'];
if cfg.save_results
    ai2_write_results(report,cfg.output_dir);
end
if cfg.make_plots
    ai2_plot_results(report,cfg.output_dir);
end
end

function cfg = defaults(cfg)
base.seed = 20260922;
base.trials_per_context = 16;
base.sensor_sigma = .08;
base.sensor_threshold = .5;
base.probe_cost = .12;
base.reward = 3;
base.action_precision = 64;
base.model_concentration = 1e8;
base.use_runtime_cache = true;
base.make_plots = true;
base.save_results = true;
base.output_dir = fullfile(fileparts(mfilename('fullpath')),'results');
names = fieldnames(base);
for k = 1:numel(names)
    if ~isfield(cfg,names{k}), cfg.(names{k}) = base.(names{k}); end
end

validateattributes(cfg.trials_per_context,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.sensor_sigma,{'numeric'},{'scalar','positive'});
validateattributes(cfg.model_concentration,{'numeric'},{'scalar','positive'});
end

function restore_state(previous_path,previous_rng,paths)
path(previous_path);
rng(previous_rng);
if ~isempty(paths.cache_root) && exist(paths.cache_root,'dir')
    rmdir(paths.cache_root,'s');
end
end
