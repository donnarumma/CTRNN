function report = test_tva_rule_task(cfg)
% Tests mathematical identities, controls, program effects and exact metrics.
if nargin<1, cfg=struct(); end
save_after=isfield(cfg,'save_results') && cfg.save_results;
cfg.save_results=false;
report=run_tva_rule_task(cfg); model=report.sensor_model; p=report.parameters;
assert(max(abs([report.summary.probability_mass]-1))<1e-12,'tva:Mass','Enumeration lost probability mass.');
assert(isequal(p,tva_parameters()),'tva:Hardware','Parameter matrices changed.');
max_bayes=0; max_matrix=0; max_compiled=0; max_identity=0; max_independence=0;
priors=[ones(1,3)/3; .1 .25 .65];
payloads={eye(3),[0 1 0;0 0 1;1 0 0]};
for m=1:2
    L=model.L(:,:,:,m);
    for ip=1:size(priors,1)
        prior=priors(ip,:);
        policy=tva_policy(prior,L,report.config.costs,'active');
        max_identity=max(max_identity,max(abs(policy.full_efe-policy.G-policy.common_constant)));
        assert(abs(policy.IG(1))<1e-12,'tva:NoopIG','Noop provides spurious information.');
        for a=1:3
            for o=1:3
                x=tva_actor(prior,L,a,o,payloads{1},p);
                y=tva_actor(prior,L,a,o,payloads{2},p);
                analytic=prior.*L(o,:,a); analytic=analytic/sum(analytic);
                max_bayes=max(max_bayes,max(abs(x.posterior-analytic)));
                max_independence=max(max_independence,max(abs(x.posterior-y.posterior)));
                % Independent dense matrix reference, including full readout.
                Q=ones(1,3)*p.Wq;
                B=[log(prior(:)) log(L(o,:,a))' zeros(3,1)];
                K=B*p.Wk; V=payloads{1}*p.Wv;
                scores=Q*K'/sqrt(3); q=exp(scores-max(scores)); q=q/sum(q);
                z=q*V; readout=dense_readout(z,p);
                max_matrix=max([max_matrix max(abs(scores-x.scores)) ...
                    max(abs(q-x.posterior)) max(abs(z-x.z)) max(abs(readout-x.pclass))]);
                max_compiled=max(max_compiled,max(abs(x.z-x.z_compiled)));
                assert(abs(sum(x.posterior)-1)<1e-12 && all(x.posterior>0),'tva:Posterior','Invalid posterior.');
                assert(abs(sum(x.z)-1)<1e-12 && all(x.z>0),'tva:Predictive','Invalid predictive mixture.');
                assert(same_ties(x.z,x.pclass,report.config.tie_tolerance), ...
                    'tva:DecisionOrder','Readout changed the maximizing class set.');
            end
        end
    end
end
assert(max_bayes<1e-12 && max_matrix<1e-12 && max_compiled<1e-12, ...
    'tva:Algebra','Bayes, matrix or compiled-reference disagreement.');
assert(max_identity<1e-12,'tva:EFE','Joint outcome EFE identity failed.');
assert(max_independence==0,'tva:PayloadLeak','Payload affects the sensor posterior.');
for m=1:2
    L=model.L(:,:,:,m); active=tva_policy(model.prior,L,model.costs,'active');
    expected=m+1;
    if isequal(model.sensor_reliabilities,[.9 .55]) && isequal(model.costs,[0 .12 .12])
        assert(active.action_probabilities(expected)==1,'tva:Mode','Known quality mode did not reverse the preferred sensor.');
    end
    ablated=tva_policy(model.prior,L,[0 .12 .12],'no_epistemic');
    assert(isequal(ablated.action_probabilities,[1 0 0]),'tva:Ablation','Ablation changed more than IG or bought costly information.');
    expensive=tva_policy(model.prior,L,[0 2 2],'active');
    assert(expensive.action_probabilities(1)==1,'tva:Cost','Expensive probes should be rejected.');
    noise=tva_policy(model.prior,ones(3,3,3)/3,[0 .12 .12],'active');
    assert(max(abs(noise.IG))<1e-12 && noise.action_probabilities(1)==1,'tva:Noise','Uninformative probes chosen.');
    certain=tva_policy([1-2e-9 1e-9 1e-9],L,[0 .12 .12],'active');
    assert(certain.action_probabilities(1)==1,'tva:Certain','Negligible uncertainty should not warrant a costly probe.');
    reliable_IG=max(active.IG(2:3));
    if reliable_IG>1e-6
        below=tva_policy(model.prior,L,[0 reliable_IG-1e-6 reliable_IG-1e-6],'active');
        above=tva_policy(model.prior,L,[0 reliable_IG+1e-6 reliable_IG+1e-6],'active');
        assert(sum(below.action_probabilities(2:3))==1 && above.action_probabilities(1)==1, ...
            'tva:Threshold','Information/cost threshold is wrong.');
    end
end
% Relabel latent rules, priors and payload rows together: class output is invariant.
L=model.L(:,:,:,1); prior=[.2 .3 .5]; perm=[3 1 2];
x=tva_actor(prior,L,2,1,eye(3),p);
I=eye(3); y=tva_actor(prior(perm),L(:,perm,:),2,1,I(perm,:),p);
assert(max(abs(x.pclass-y.pclass))<1e-12,'tva:RuleRelabel','Rule relabeling changed class predictions.');
% The actor's behavior is determined by available inputs; hidden truth is only
% a field of enumeration/scoring, never part of tva_actor or tva_policy APIs.
x2=tva_actor(prior,L,2,1,eye(3),p);
assert(isequal(x,x2),'tva:HiddenInput','Identical observable inputs produced different outputs.');
% A direct program swap changes execution with identical payload/hardware.
f1=tva_actor(prior,L,2,1,eye(3),p,1); f2=tva_actor(prior,L,2,1,eye(3),p,2);
assert(isequal(f1.posterior,f2.posterior) && max(abs(f1.z-f2.z))==1, ...
    'tva:ProgramEffect','Program swap does not isolate execution.');
% Readout acts on each row independently and the FFN has a nonzero effect.
batch=tva_readout([x.z;y.z;ones(1,3)/3],p);
one=tva_readout(x.z,p); assert(max(abs(batch.pclass(1,:)-one.pclass))<1e-12,'tva:ReadoutRows','Readout mixes tokens.');
assert(max(abs(one.ffn))>.01,'tva:FFN','FFN is accidentally inactive.');
uniform=tva_readout(ones(1,3)/3,p);
assert(max(abs(uniform.pclass-ones(1,3)/3))<1e-10,'tva:Ties','Uniform belief is biased by floating-point tie breaking.');
% Actual outcome rows show all three hidden truths under the same action policy.
for m=1:2
    for c=1:3
        selected=report.trials(strcmp({report.trials.condition},'active') & [report.trials.mode]==m & [report.trials.truth]==c);
        a_mass=zeros(1,3);
        for a=1:3, a_mass(a)=sum([selected([selected.action]==a).mass]); end
        a_mass=a_mass/sum(a_mass);
        policy=tva_policy(model.prior,model.L(:,:,:,m),model.costs,'active');
        assert(max(abs(a_mass-policy.action_probabilities))<1e-12,'tva:PolicyLeak','Action policy depends on hidden truth.');
    end
end
if isequal(model.sensor_reliabilities,[.9 .55]) && isequal(model.costs,[0 .12 .12])
    active=lookup(report,'active'); random=lookup(report,'random_probe'); noep=lookup(report,'no_epistemic');
    passive=lookup(report,'passive'); oracle=lookup(report,'oracle'); compiled=lookup(report,'compiled_matched');
    assert(abs(active.accuracy-.9)<1e-12 && abs(random.accuracy-.725)<1e-12 && abs(noep.accuracy-1/3)<1e-12, ...
        'tva:ExpectedAccuracy','Exact expected accuracies differ from the analytical values.');
    assert(abs(passive.accuracy-1/3)<1e-12 && abs(oracle.accuracy-1)<1e-12,'tva:Controls','Passive/oracle failed.');
    assert(abs(active.cost-random.cost)<1e-12 && abs(active.paid_probe-random.paid_probe)<1e-12, ...
        'tva:MatchedBudget','Active and random probe budgets differ.');
    assert(abs(active.accuracy-compiled.accuracy)<1e-12 && abs(active.classhead_nll-compiled.classhead_nll)<1e-12, ...
        'tva:Compiled','Matched compiled bank differs from interpreted execution.');
    for name={'fixed_program','fixed_program_2','fixed_program_3'}
        fixed=lookup(report,name{1}); assert(abs(fixed.accuracy-1/3)<1e-12,'tva:Fixed','Uniform-prior fixed selectors should tie.');
    end
    assert(active.classhead_nll>active.bayes_predictive_nll+1e-3, ...
        'tva:Calibration','Decision head is being mistaken for the exact predictive distribution.');
end
report.validation=struct('passed',true,'max_bayes_error',max_bayes, ...
    'max_scalar_matrix_error',max_matrix,'max_compiled_error',max_compiled, ...
    'max_efe_identity_error',max_identity,'max_payload_posterior_change',max_independence, ...
    'exact_probability_mass',true,'epistemic_ablation',true,'known_mode_sensor_reversal',true, ...
    'fixed_parameters',true,'program_intervention',true,'classhead_is_not_predictive_mixture',true);
if report.config.use_spm_reference
    without_cfg=report.config; without_cfg.use_spm_reference=false; without_cfg.save_results=false;
    without=run_tva_rule_task(without_cfg);
    assert(isequal(report.summary,without.summary) && isequal(report.trials,without.trials) && ...
        isequal(report.cost_sweep,without.cost_sweep) && isequal(report.examples,without.examples), ...
        'tva:SPMLeak','Optional SPM reference changed primary results.');
    assert(report.spm.maxerror<1e-10,'tva:SPM','SPM reference disagrees with attention.');
    report.validation.spm_reference_independence=true;
end
if save_after, tva_write_results(report,report.config.output_dir); end
fprintf('PASS: exact enumeration, Bayes attention, EFE identity, controls, budget, fixed programs and readout.\n');
end

function pclass=dense_readout(z,p)
ln=@(T,g,b) ((T-mean(T,2))./sqrt(mean((T-mean(T,2)).^2,2)+p.layernorm_epsilon)).*g+b;
U=ln(ones(size(z))+z*p.Wout,p.gamma1,p.beta1);
ffn=max(0,U*p.W1+p.b1)*p.W2+p.b2;
Y=ln(U+ffn,p.gamma2,p.beta2); logits=Y*p.Wclass+p.bclass;
e=exp(logits-max(logits,[],2)); pclass=e./sum(e,2);
end

function same=same_ties(a,b,tol)
same=isequal(abs(a-max(a))<=tol,abs(b-max(b))<=tol);
end

function s=lookup(report,name)
s=report.summary(strcmp({report.summary.condition},name));
end
