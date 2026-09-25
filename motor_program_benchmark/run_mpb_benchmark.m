function report = run_mpb_benchmark(cfg)
%RUN_MPB_BENCHMARK Exact finite motor-program proof of concept, protocol v1.
% Full run: report=run_mpb_benchmark(); plots/data are saved under results/.
% Smoke: run_mpb_benchmark(struct('quick',true,'save_results',false,'make_plots',false)).
if nargin<1, cfg=struct(); end
if ~isfield(cfg,'quick'), cfg.quick=false; end
if ~isfield(cfg,'save_results'), cfg.save_results=true; end
if ~isfield(cfg,'make_plots'), cfg.make_plots=true; end
if ~isfield(cfg,'run_tests'), cfg.run_tests=true; end
if ~isfield(cfg,'use_spm_reference'), cfg.use_spm_reference=true; end
if ~isfield(cfg,'output_dir'), cfg.output_dir=fullfile(fileparts(mfilename('fullpath')),'results'); end
c=mpb_protocol();
if cfg.quick, c.n_development=1; c.n_evaluation=2; c.n_transfer=1; end
p=mpb_parameters(); initial_parameters=p;
report=struct(); report.metadata=struct('protocol',c.version,'quick',cfg.quick, ...
    'runtime',version(),'platform',computer(),'enumeration','Exact route and sensory-outcome probabilities', ...
    'learning',false,'ctrnn_realization',false,'spm_in_policy',false, ...
    'motor_protocol','One optional LOOK then committed program; closed-loop GO uses observed position.', ...
    'uncertainty_statement','Exact means on a finite layout suite; no binomial confidence intervals.');
report.protocol=c; report.parameters=p; report.config=cfg;
if cfg.run_tests
    report.tests.core=test_mpb_core(); report.tests.policy=test_mpb_policy();
    report.tests.integration=test_mpb_benchmark();
end
layouts=mpb_layouts(c); report.layouts=layouts;
Lprobe=likelihood(c.probe_reliability);
[beliefs,belief_rows,bstats]=belief_cache(c,p,Lprobe);
report.belief_checks=belief_rows;
report.core_stats=bstats; report.core_stats.max_compiled_error=0;
report.core_stats.max_ideal_error=0; report.core_stats.max_cost_prediction_error=0;
report.core_stats.max_branch_mass_error=0;
report.layout_metrics=struct('layout',{},'split',{},'regime',{},'condition',{},'values',{},'mass',{});
report.route_metrics=struct('layout',{},'split',{},'regime',{},'condition',{},'route',{},'composition',{},'values',{});
report.cost_sweep_layouts=struct('layout',{},'regime',{},'condition',{},'look_cost',{},'values',{});
report.examples=struct('label',{},'layout',{},'program',{},'truth',{},'trace',{});
backends={'interpreted','compiled','frozen_keys','frozen_attention','reset_memory'};
for li=1:numel(layouts)
    layout=layouts(li); fprintf('Layout %d/%d: %s\n',li,numel(layouts),layout.id);
    [terminal,execution_cost,predicted_steps]=mpb_predict_costs(layout,p,c);
    cache=cell(5,6,6);
    for bi=1:5
        for selected=1:6
            for truth=1:6
                m=mpb_rollout(layout,selected,truth,backends{bi},p,c,false);
                cache{bi,selected,truth}=m;
                report.core_stats.max_compiled_error=max(report.core_stats.max_compiled_error,m.max_compiled_error);
                report.core_stats.max_ideal_error=max(report.core_stats.max_ideal_error,m.max_ideal_error);
                if bi<=2
                    assert(m.success==double(terminal(selected,truth)==1),'mpb:OutcomeModel','Execution differs from declared terminal model.');
                    assert(m.motor_steps==predicted_steps(selected,truth),'mpb:StepModel','Motor length differs from analytic model.');
                    report.core_stats.max_cost_prediction_error=max(report.core_stats.max_cost_prediction_error,abs(m.motor_cost-execution_cost(selected,truth)));
                end
            end
        end
    end
    if strcmp(layout.split,'evaluation') && isempty(report.examples)
        indices=[1 3 5]; labels={'interpreted_correct','frozen_keys','reset_memory'};
        for ex=1:3
            m=mpb_rollout(layout,6,6,backends{indices(ex)},p,c,true);
            report.examples(end+1)=struct('label',labels{ex},'layout',layout, ...
                'program',p.routes(6,:),'truth',p.routes(6,:),'trace',m.trace); %#ok<AGROW>
        end
    end
    policy_bank=cell(numel(c.initial_reliability),2,6); pooled_look_rate=0;
    for ri=1:numel(c.initial_reliability)
        b=beliefs(ri);
        for mode=1:2
            mode_name='active'; if mode==2, mode_name='no_epistemic'; end
            for cue=1:6
                pol=mpb_policy(b.initial(cue,:),Lprobe,terminal,execution_cost, ...
                    c.look_cost,c.failure_penalty,mode_name);
                policy_bank{ri,mode,cue}=pol;
                if mode==1, pooled_look_rate=pooled_look_rate+double(pol.use_look)/(6*numel(c.initial_reliability)); end
            end
        end
    end
    for ri=1:numel(c.initial_reliability)
        b=beliefs(ri); policies=reshape(policy_bank(ri,:,:),2,6);
        for ci=1:numel(c.conditions)
            condition=c.conditions{ci};
            [values,per_route,mass]=evaluate_condition(condition,policies,b,cache,Lprobe,c.look_cost,c,pooled_look_rate);
            report.core_stats.max_branch_mass_error=max(report.core_stats.max_branch_mass_error,max(abs(mass-1)));
            report.layout_metrics(end+1)=struct('layout',layout.id,'split',layout.split, ...
                'regime',c.regime_names{ri},'condition',condition,'values',values,'mass',mean(mass)); %#ok<AGROW>
            for truth=1:6
                composition='reference'; if ismember(truth,c.withheld_programs), composition='withheld'; end
                report.route_metrics(end+1)=struct('layout',layout.id,'split',layout.split, ...
                    'regime',c.regime_names{ri},'condition',condition,'route',truth, ...
                    'composition',composition,'values',per_route(truth,:)); %#ok<AGROW>
            end
        end
        if strcmp(layout.split,'evaluation')
            for cost=c.cost_sweep
                for mode=1:2
                    condition='active'; if mode==2, condition='no_epistemic'; end
                    [values,~,mass]=evaluate_condition(condition,policies,b,cache,Lprobe,cost,c);
                    assert(max(abs(mass-1))<1e-12,'mpb:Mass','Sweep probability mass differs from one.');
                    report.cost_sweep_layouts(end+1)=struct('layout',layout.id, ...
                        'regime',c.regime_names{ri},'condition',condition,'look_cost',cost,'values',values); %#ok<AGROW>
                end
            end
        end
    end
    assert(isequal(p,initial_parameters),'mpb:WeightsChanged','Fixed parameters changed during evaluation.');
end
report.summary=aggregate_layouts(report.layout_metrics,c);
report.cost_sweep=aggregate_sweep(report.cost_sweep_layouts,c);
report.structural_counts=struct('shared_token_rows',p.n_tokens,'features_per_token',p.d_model, ...
    'query_key_width',p.d_k,'value_width',p.d_v,'projection_dense_entries',numel(p.Wq)+numel(p.Wk)+numel(p.Wv), ...
    'projection_nonzeros',nnz(p.Wq)+nnz(p.Wk)+nnz(p.Wv),'readout_nonzeros',nnz(p.Rbelief)+nnz(p.Rmotor), ...
    'runtime_program_matrix_entries',16,'compiled_selector_bank_entries',6*4*4, ...
    'primitive_controllers',4,'programs',6, ...
    'interpretation','Arithmetic storage counts; not neuron counts or a neural-efficiency claim.');
report.validation=validate_report(report);
if cfg.use_spm_reference, report.spm=mpb_spm_reference(report,cfg); end
if cfg.save_results
    mpb_write_results(report,cfg.output_dir);
    if cfg.make_plots, mpb_plot_results(report,cfg.output_dir); end
end
end

function L=likelihood(reliability)
L=ones(6)*(1-reliability)/5; L(1:7:end)=reliability;
end

function [b,rows,stats]=belief_cache(c,p,Lprobe)
b=struct('Linitial',{},'initial',{},'after',{});
rows=struct('regime',{},'initial_cue',{},'probe_cue',{},'prior',{},'posterior',{},'reference',{},'max_error',{},'F_before',{},'F_after',{});
stats=struct('max_bayes_error',0,'max_F_increase',0,'max_F_identity_error',0);
prior=ones(1,6)/6;
for ri=1:numel(c.initial_reliability)
    L=likelihood(c.initial_reliability(ri)); initial=zeros(6); after=zeros(6,6,6);
    for cue=1:6
        out=mpb_belief_update(prior,L,cue,p); initial(cue,:)=out.posterior;
        [rows,stats]=record_belief(rows,stats,out,c.regime_names{ri},cue,0,prior,L);
        for probe=1:6
            child=mpb_belief_update(out.posterior,Lprobe,probe,p);
            after(cue,probe,:)=child.posterior;
            [rows,stats]=record_belief(rows,stats,child,c.regime_names{ri},cue,probe,out.posterior,Lprobe);
        end
    end
    b(ri)=struct('Linitial',L,'initial',initial,'after',after);
end
end

function [rows,stats]=record_belief(rows,stats,out,regime,cue,probe,prior,L)
obs=cue; if probe>0, obs=probe; end
ref=prior.*L(obs,:); ref=ref/sum(ref);
err=max(abs(ref-out.posterior));
stats.max_bayes_error=max(stats.max_bayes_error,err);
stats.max_F_increase=max(stats.max_F_increase,out.F_at_posterior-out.F_at_prior);
stats.max_F_identity_error=max(stats.max_F_identity_error,abs(out.F_at_posterior+log(sum(prior.*L(obs,:)))));
rows(end+1)=struct('regime',regime,'initial_cue',cue,'probe_cue',probe, ...
    'prior',prior,'posterior',out.posterior,'reference',ref,'max_error',err, ...
    'F_before',out.F_at_prior,'F_after',out.F_at_posterior);
end

function [values,per_route,mass]=evaluate_condition(condition,policies,b,cache,Lprobe,look_cost,c,pooled_look_rate)
if nargin<8, pooled_look_rate=NaN; end
per_route=zeros(6,numel(c.metrics)); mass=zeros(1,6);
mode=1; if strcmp(condition,'no_epistemic'), mode=2; end
backend=1;
if strcmp(condition,'compiled_matched'), backend=2; end
if strcmp(condition,'frozen_motor_keys'), backend=3; end
if strcmp(condition,'frozen_motor_attention'), backend=4; end
if strcmp(condition,'reset_memory'), backend=5; end
for cue=1:6
    pol=policies{mode,cue}; [choice,commit]=select(pol,look_cost-c.look_cost);
    prob_look=double(choice==7);
    if strcmp(condition,'no_look'), prob_look=0; end
    if strcmp(condition,'matched_random_look')
        assert(isfinite(pooled_look_rate),'mpb:Budget','Pooled LOOK rate must be supplied.');
        prob_look=pooled_look_rate;
    end
    if strcmp(condition,'oracle'), prob_look=0; end
    for truth=1:6
        base=b.Linitial(cue,truth);
        route=commit; posterior=b.initial(cue,:);
        if strcmp(condition,'oracle'), route=truth; posterior=zeros(1,6); posterior(truth)=1; end
        w=base*(1-prob_look);
        if w>0
            per_route(truth,:)=per_route(truth,:)+w*branch_metrics(cache{backend,route,truth},posterior,truth,0,look_cost,c);
            mass(truth)=mass(truth)+w;
        end
        if prob_look>0
            for probe=1:6
                route=pol.contingent_routes(probe);
                posterior=reshape(b.after(cue,probe,:),1,6);
                w=base*prob_look*Lprobe(probe,truth);
                per_route(truth,:)=per_route(truth,:)+w*branch_metrics(cache{backend,route,truth},posterior,truth,1,look_cost,c);
                mass(truth)=mass(truth)+w;
            end
        end
    end
end
assert(max(abs(mass-1))<1e-12,'mpb:Mass','Conditional route mass differs from one.');
values=mean(per_route,1);
end

function [choice,commit]=select(pol,cost_delta)
G=pol.policy_G; G(7)=G(7)+cost_delta;
choice=find(G<=min(G)+1e-12,1); commit=find(G(1:6)<=min(G(1:6))+1e-12,1);
end

function v=branch_metrics(m,b,truth,look,look_cost,c)
onehot=zeros(1,6); onehot(truth)=1;
pos=b>0; entropy=-sum(b(pos).*log(b(pos)));
total=m.motor_cost+look*look_cost;
v=[m.success,m.wrong_goal,m.moves,m.touches,m.motor_steps,look, ...
    m.motor_cost,total,c.failure_penalty*(1-m.success)+total, ...
    sum((b-onehot).^2),-log(max(realmin,b(truth))),entropy,m.timeout];
end

function summary=aggregate_layouts(rows,c)
summary=struct('split',{},'regime',{},'condition',{},'n_layouts',{},'values',{});
splits={'development','evaluation','geometry_transfer'};
for si=1:3
    for ri=1:numel(c.regime_names)+1
        if ri>numel(c.regime_names), regime='mixture'; else, regime=c.regime_names{ri}; end
        for ci=1:numel(c.conditions)
            match=strcmp({rows.split},splits{si}) & strcmp({rows.condition},c.conditions{ci});
            if ~strcmp(regime,'mixture'), match=match & strcmp({rows.regime},regime); end
            vals=vertcat(rows(match).values);
            summary(end+1)=struct('split',splits{si},'regime',regime, ...
                'condition',c.conditions{ci},'n_layouts',numel(unique({rows(match).layout})),'values',mean(vals,1)); %#ok<AGROW>
        end
    end
end
end

function summary=aggregate_sweep(rows,c)
summary=struct('regime',{},'condition',{},'look_cost',{},'n_layouts',{},'values',{});
conditions={'active','no_epistemic'};
for ri=1:numel(c.regime_names)
    for ci=1:2
        for cost=c.cost_sweep
            match=strcmp({rows.regime},c.regime_names{ri}) & strcmp({rows.condition},conditions{ci}) & abs([rows.look_cost]-cost)<1e-12;
            vals=vertcat(rows(match).values);
            summary(end+1)=struct('regime',c.regime_names{ri},'condition',conditions{ci}, ...
                'look_cost',cost,'n_layouts',sum(match),'values',mean(vals,1)); %#ok<AGROW>
        end
    end
end
end

function v=validate_report(r)
rows=r.layout_metrics; max_compiled=0; max_budget=0; max_nll_entropy=0;
for k=1:numel(rows)
    if strcmp(rows(k).condition,'active')
        same=strcmp({rows.layout},rows(k).layout) & strcmp({rows.regime},rows(k).regime);
        comp=find(same & strcmp({rows.condition},'compiled_matched'));
        random=find(same & strcmp({rows.condition},'matched_random_look'));
        max_compiled=max(max_compiled,max(abs(rows(k).values-rows(comp).values)));
        %#ok<NASGU> Random LOOK is budget-matched over the sensor-regime mixture.
    end
    max_nll_entropy=max(max_nll_entropy,abs(rows(k).values(11)-rows(k).values(12)));
end
ids=unique({rows.layout});
for k=1:numel(ids)
    a=strcmp({rows.layout},ids{k}) & strcmp({rows.condition},'active');
    b=strcmp({rows.layout},ids{k}) & strcmp({rows.condition},'matched_random_look');
    av=vertcat(rows(a).values); bv=vertcat(rows(b).values);
    max_budget=max(max_budget,abs(mean(av(:,6))-mean(bv(:,6))));
end
assert(max_compiled<1e-12,'mpb:CompiledMismatch','Compiled and interpreted metrics differ.');
assert(max_budget<1e-12,'mpb:BudgetMismatch','Random LOOK baseline has a different expected sensing budget.');
assert(max_nll_entropy<1e-12,'mpb:Calibration','Exact expected log loss differs from posterior entropy.');
assert(r.core_stats.max_cost_prediction_error<1e-12,'mpb:Model','Analytic and executed costs differ.');
assert(r.core_stats.max_bayes_error<1e-12,'mpb:Bayes','Attention does not reproduce Bayes.');
assert(r.core_stats.max_F_increase<1e-12,'mpb:FreeEnergy','The posterior raises current-data free energy.');
v=struct('max_compiled_metric_difference',max_compiled,'max_matched_LOOK_budget_difference',max_budget, ...
    'max_expected_logloss_entropy_difference',max_nll_entropy,'passed',true);
end
