function report = run_bayesian_circuit(cfg)
% Hypothesis B1: neural Bayesian recognition under a model supplied as code.
% One observation per reset, uniform prior; no neural memory or action choice.
if nargin<1, cfg=struct(); end
cfg = defaults(cfg);
old_path=path; old_rng=rng;
paths=bci_setup(cfg);
guard=onCleanup(@() cleanup(paths,old_path,old_rng)); %#ok<NASGU>
anchors=[.8 .2;.6 .4;.2 .8;.75 .4];
names={'strong','weak','reversed','asymmetric'};
[p1,p2]=ndgrid(cfg.grid_levels,cfg.grid_levels);
models=[anchors; p1(:) p2(:)];
bank=bci_build_circuits(models,cfg);
fprintf('B1: %d model configurations, 2 observations, one fixed %d-neuron interpreter.\n', ...
    size(models,1),bank.interpreter.numUnits);
observations=eye(2);
episodes=struct([]); examples=struct([]); index=0;
for m=1:size(models,1)
    for o=1:2
        target=bci_simulate(bank.targets{m},observations(o,:),[],cfg);
        interpreted=bci_simulate(bank.interpreter,observations(o,:),bank.programs(m,:),cfg);
        if o==1, likelihood=models(m,:); else, likelihood=1-models(m,:); end
        measured=bci_measure(target,interpreted,likelihood,bank.prior);
        index=index+1;
        if m<=4, group='anchor'; name=names{m}; else, group='grid'; name=sprintf('grid_%02d',m-4); end
        episodes(index)=scalar_episode(measured,m,o,group,name,models(m,:)); %#ok<AGROW>
        if m<=4
            examples(index).model=m; examples(index).observation=o; %#ok<AGROW>
            examples(index).time=[0;target.time];
            examples(index).raw=[bank.prior;interpreted.readout];
            examples(index).target=[bank.prior;target.readout];
            examples(index).diagnostics=measured;
        end
    end
end
% SPM is evaluated AFTER the primary neural rollouts, solely as an independent reference.
if cfg.run_spm_reference
    fprintf('Checking the same static posteriors with the original SPM solver...\n');
    for k=1:numel(episodes)
        e=episodes(k);
        [q,info]=bci_spm_reference(e.likelihood_high,e.observation,bank.prior');
        episodes(k).spm_posterior=q';
        episodes(k).spm_error=max(abs(q'-e.bayes));
    end
    report.spm_reference_info=info;
end
report.config=cfg; report.paths=paths; report.runtime=version;
report.models=models; report.model_names=names; report.programs=bank.programs;
report.weights=bank.weights; report.episodes=episodes; report.examples=examples;
report.hardware=struct('W',bank.interpreter.internalWMatrix,'V',bank.interpreter.externalWMatrix, ...
    'biases',bank.interpreter.biases,'tau',bank.interpreter.tau, ...
    'n_neurons',bank.interpreter.numUnits,'fixed',bank.fixed_hardware);
report.summary=struct('n_observations',numel(episodes),'n_grid_models',numel(p1), ...
    'max_raw_error',max([episodes.raw_error]),'max_decoded_error',max([episodes.decoded_error]), ...
    'max_normalization_error',max([episodes.normalization_error]), ...
    'max_final_kl',max([episodes.final_kl]),'max_target_error',max([episodes.target_error]), ...
    'max_spm_error',max([episodes.spm_error]), ...
    'max_free_energy_increase',max([episodes.max_free_energy_increase]), ...
    'max_target_free_energy_increase',max([episodes.target_max_free_energy_increase]));
if cfg.run_sensitivity
    fprintf('Checking time step and encoding range on the four anchor models...\n');
    fine_cfg=cfg; fine_cfg.dt=cfg.dt/2;
    narrow_cfg=cfg; narrow_cfg.weight_range=[-2 2];
    fine=bci_build_circuits(anchors,fine_cfg);
    narrow=bci_build_circuits(anchors,narrow_cfg);
    for m=1:4
        for o=1:2
            k=2*(m-1)+o;
            f=bci_simulate(fine.interpreter,observations(o,:),fine.programs(m,:),fine_cfg);
            n=bci_simulate(narrow.interpreter,observations(o,:),narrow.programs(m,:),narrow_cfg);
            report.sensitivity.half_dt_change(k)=max(abs(f.readout(end,:)-episodes(k).raw_final));
            report.sensitivity.narrow_raw_error(k)=max(abs(n.readout(end,:)-episodes(k).bayes));
        end
    end
    report.sensitivity.narrow_weight_range=[-2 2];
end
fprintf('Max raw posterior error %.6f; normalization defect %.6f; diagnostic KL %.6f nat.\n', ...
    report.summary.max_raw_error,report.summary.max_normalization_error,report.summary.max_final_kl);
fprintf('Analytic target error %.3g; SPM error %.3g; largest interpreted F increase/step %.3g.\n', ...
    report.summary.max_target_error,report.summary.max_spm_error,report.summary.max_free_energy_increase);
if cfg.save_results, bci_write_results(report,cfg.output_dir); end
if cfg.make_plots, bci_plot_results(report,cfg.output_dir); end
end

function e=scalar_episode(m,index,o,group,name,p)
fields={'bayes','raw_final','raw_steady','target_final','decoded_final','raw_error', ...
    'target_error','decoded_error','normalization_error','final_normalization_error', ...
    'tail_range','emulation_rmse','final_kl','max_free_energy_increase', ...
    'total_free_energy_increase','target_max_free_energy_increase'};
e=struct('model',index,'name',name,'group',group,'observation',o,'likelihood_high',p);
for k=1:numel(fields), e.(fields{k})=m.(fields{k}); end
e.spm_posterior=[NaN NaN]; e.spm_error=NaN;
end

function cfg=defaults(cfg)
base=struct('dt',.05,'duration',60,'target_tau',5,'multiplier_tau',.25, ...
    'weight_range',[-5 5],'grid_levels',[.2 .35 .5 .65 .8], ...
    'run_spm_reference',true,'run_sensitivity',true,'use_runtime_cache',true, ...
    'save_results',true,'make_plots',true, ...
    'output_dir',fullfile(fileparts(mfilename('fullpath')),'results'));
f=fieldnames(base);
for k=1:numel(f), if ~isfield(cfg,f{k}), cfg.(f{k})=base.(f{k}); end; end
validateattributes(cfg.dt,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.duration,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.target_tau,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.multiplier_tau,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.weight_range,{'numeric'},{'row','numel',2,'real','finite'});
assert(cfg.weight_range(1)<cfg.weight_range(2),'bci:Range','Invalid weight interval.');
validateattributes(cfg.grid_levels,{'numeric'},{'vector','nonempty','real','finite','positive'});
assert(all(cfg.grid_levels<1),'bci:Grid','Grid likelihoods must be in (0,1).');
end

function cleanup(paths,old_path,old_rng)
path(old_path); rng(old_rng);
if ~isempty(paths.cache_root) && exist(paths.cache_root,'dir'), rmdir(paths.cache_root,'s'); end
end
