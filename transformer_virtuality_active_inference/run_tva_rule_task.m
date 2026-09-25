function report = run_tva_rule_task(cfg)
% Exact finite enumeration: active sensing -> belief program -> payload readout.
% report = run_tva_rule_task(struct('save_results',false))
% No Monte Carlo, training, neural multiplier approximation, or hidden-rule
% argument is used by the actor. SPM is an optional independent reference.
if nargin<1, cfg=struct(); end
cfg=defaults(cfg); model=tva_model(cfg); p=tva_parameters();
conditions={'active','no_epistemic','passive','random_probe', ...
    'fixed_program','fixed_program_2','fixed_program_3','oracle','compiled_matched'};
all_trials=struct([]); summaries=struct([]); examples=struct([]);
for k=1:numel(conditions)
    condition=conditions{k}; rows=struct([]);
    for m=1:2
        L=model.L(:,:,:,m);
        for perm_index=1:size(model.payload_permutations,1)
            identity=eye(3); payload=identity(model.payload_permutations(perm_index,:),:);
            for truth=1:3
                prior=model.prior;
                if strcmp(condition,'oracle')
                    prior=cfg.oracle_epsilon*ones(1,3); prior(truth)=1-2*cfg.oracle_epsilon;
                end
                policy=tva_policy(prior,L,cfg.costs,condition);
                selector=[];
                if strcmp(condition,'fixed_program'), selector=1;
                elseif strcmp(condition,'fixed_program_2'), selector=2;
                elseif strcmp(condition,'fixed_program_3'), selector=3; end
                for a=1:3
                    pa=policy.action_probabilities(a);
                    if pa==0, continue; end
                    for o=1:3
                        out=tva_actor(prior,L,a,o,payload,p,selector);
                        if strcmp(condition,'compiled_matched')
                            out.z=out.z_compiled;
                            out.readout=tva_readout(out.z,p); out.pclass=out.readout.pclass;
                        end
                        [~,true_class]=max(payload(truth,:));
                        mass=(1/2)*(1/6)*model.prior(truth)*pa*L(o,truth,a);
                        predicted=tie_set(out.pclass,cfg.tie_tolerance);
                        code_choice=tie_set(out.program,cfg.tie_tolerance);
                        row=struct('condition',condition,'mode',m,'permutation',perm_index, ...
                            'truth',truth,'action',a,'observation',o,'mass',mass, ...
                            'true_class',true_class,'accuracy',double(predicted(true_class))/sum(predicted), ...
                            'rule_accuracy',double(code_choice(truth))/sum(code_choice), ...
                            'paid_probe',double(a>1),'cost',cfg.costs(a), ...
                            'posterior_true',out.posterior(truth), ...
                            'program_true',out.program(truth), ...
                            'posterior_entropy',-sum(out.posterior.*log(out.posterior)), ...
                            'classhead_nll',-log(out.pclass(true_class)), ...
                            'bayes_predictive_nll',-log(out.bayes_predictive(true_class)), ...
                            'execution_mse',mean((out.z-payload(truth,:)).^2), ...
                            'q',out.posterior,'program',out.program,'z',out.z, ...
                            'bayes_predictive',out.bayes_predictive,'pclass',out.pclass);
                        rows(end+1)=row; %#ok<AGROW>
                        if k==1 && perm_index==1 && truth==1 && o==1
                            examples(m)=struct('mode',m,'payload',payload,'policy',policy,'actor',out); %#ok<AGROW>
                        end
                    end
                end
            end
        end
    end
    summary=aggregate(rows,condition);
    summaries(k)=summary; all_trials=[all_trials rows]; %#ok<AGROW>
end
report=struct('config',cfg,'parameters',p,'sensor_model',model, ...
    'summary',summaries,'trials',all_trials,'examples',examples);
report.cost_sweep=cost_sweep(cfg,model);
report.posterior_checks=posterior_checks(model,p,cfg);
report.metadata=struct('runtime',version,'n_hidden_rules',3,'n_payload_features',3, ...
    'n_payload_permutations',6,'n_known_sensor_modes',2,'n_sensor_outcomes',3, ...
    'evaluation','exact finite enumeration, not sampling', ...
    'program_representation','posterior probabilities over three row selectors', ...
    'hardware','fixed arithmetic operators and fixed hand-chosen matrices; no CTRNN neuron count', ...
    'classhead','decision readout; z is exact Bayesian predictive only without a fixed-program override', ...
    'oracle','near-point prior supplied as an explicit privileged-information control', ...
    'training',false,'reward_preferences','uniform observations and cost-category preferences; epistemic special case');
if cfg.use_spm_reference
    assert(exist('tva_spm_reference','file')~=0,'tva:SPMReference', ...
        'Install tva_spm_reference.m or set use_spm_reference=false.');
    report.spm=tva_spm_reference(report.posterior_checks,cfg);
end
if cfg.save_results, tva_write_results(report,cfg.output_dir); end
end

function s=aggregate(rows,condition)
w=[rows.mass];
s=struct('condition',condition,'probability_mass',sum(w),'enumerated_branches',numel(rows));
metrics={'accuracy','rule_accuracy','paid_probe','cost','posterior_true','program_true', ...
    'posterior_entropy','classhead_nll','bayes_predictive_nll','execution_mse'};
for k=1:numel(metrics), s.(metrics{k})=sum(w.*[rows.(metrics{k})]); end
end

function chosen=tie_set(scores,tolerance)
chosen=abs(scores-max(scores))<=tolerance;
end

function sweep=cost_sweep(cfg,model)
sweep=struct([]);
for cost=cfg.cost_grid
    for m=1:2
        for entry=1:2
            conditions={'active','no_epistemic'}; condition=conditions{entry};
            policy=tva_policy(model.prior,model.L(:,:,:,m),[0 cost cost],condition);
            accuracy=0;
            for a=1:3
                for o=1:3
                    joint=model.prior.*model.L(o,:,a,m);
                    % Bayes-optimal classification under one-to-one payload labels.
                    accuracy=accuracy+policy.action_probabilities(a)*max(joint);
                end
            end
            sweep(end+1)=struct('cost',cost,'mode',m,'condition',condition, ...
                'noop_probability',policy.action_probabilities(1), ...
                'left_probability',policy.action_probabilities(2), ...
                'right_probability',policy.action_probabilities(3), ...
                'expected_accuracy',accuracy,'expected_paid_cost',cost*sum(policy.action_probabilities(2:3)), ...
                'left_IG',policy.IG(2),'right_IG',policy.IG(3)); %#ok<AGROW>
        end
    end
end
end

function checks=posterior_checks(model,p,cfg)
checks=struct([]); priors=[model.prior; cfg.oracle_epsilon*ones(3)+(1-3*cfg.oracle_epsilon)*eye(3)];
for m=1:2
    for a=1:3
        for o=1:3
            for k=1:size(priors,1)
                out=tva_actor(priors(k,:),model.L(:,:,:,m),a,o,eye(3),p);
                checks(end+1)=struct('likelihood',model.L(:,:,a,m),'prior',priors(k,:)', ...
                    'observationindex',o,'posterioractual',out.posterior); %#ok<AGROW>
            end
        end
    end
end
end

function cfg=defaults(cfg)
if isfield(cfg,'run_spm_reference') && ~isfield(cfg,'use_spm_reference')
    cfg.use_spm_reference=cfg.run_spm_reference;
end
base=struct('sensor_reliabilities',[.9 .55],'costs',[0 .12 .12], ...
    'oracle_epsilon',1e-9,'tie_tolerance',1e-10,'cost_grid',[0 .02 .06 .12 .3 .5 .7 .704214597220667 .72 1], ...
    'use_spm_reference',false,'save_results',true, ...
    'output_dir',fullfile(fileparts(mfilename('fullpath')),'results'));
names=fieldnames(base);
for k=1:numel(names), if ~isfield(cfg,names{k}), cfg.(names{k})=base.(names{k}); end; end
assert(isscalar(cfg.oracle_epsilon) && cfg.oracle_epsilon>0 && cfg.oracle_epsilon<1/3, ...
    'tva:OraclePrior','Oracle epsilon must be positive and below one third.');
assert(isscalar(cfg.tie_tolerance) && cfg.tie_tolerance>0 && isfinite(cfg.tie_tolerance), ...
    'tva:Ties','Tie tolerance must be positive.');
assert(isvector(cfg.cost_grid) && all(isfinite(cfg.cost_grid)) && all(cfg.cost_grid>=0), ...
    'tva:CostGrid','Cost sweep values must be nonnegative finite values.');
cfg.cost_grid=cfg.cost_grid(:)';
end
