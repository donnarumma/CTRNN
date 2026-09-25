function report = test_bayesian_circuit(cfg)
% Verify the implemented B1 scope; approximation thresholds are empirical.
if nargin<1, cfg=struct(); end
cfg.save_results=false; cfg.make_plots=false;
cfg.run_spm_reference=true; cfg.run_sensitivity=true;
report=run_bayesian_circuit(cfg);
s=report.summary;
assert(report.hardware.fixed && report.hardware.n_neurons==14,'bci:Hardware','Wrong interpreter.');
assert(s.max_target_error<1e-5,'bci:Analytic','Dedicated circuit does not converge to Bayes.');
assert(s.max_spm_error<1e-10,'bci:SPM','SPM disagrees with analytic Bayes.');
assert(s.max_target_free_energy_increase<1e-12,'bci:TargetF','Ideal circuit increases free energy.');
assert(s.max_raw_error<.08,'bci:Approximation','Raw posterior error exceeds prototype tolerance.');
assert(s.max_normalization_error<.12,'bci:Normalization','Raw complementarity defect too large.');
assert(s.max_final_kl<.015,'bci:KL','Diagnostic posterior KL too large.');
assert(max(report.sensitivity.half_dt_change)<.001,'bci:Euler','Time step materially changes endpoint.');
e=report.episodes;
assert(e(1).raw_final(1)>e(3).raw_final(1)+.1,'bci:ProgramEffect','Changing code has too little effect.');
assert(e(1).raw_final(1)>e(5).raw_final(1)+.4,'bci:Reversal','Reversing model fails to reverse inference.');
for k=1:numel(e)
    assert(all(e(k).raw_final>0 & e(k).raw_final<1),'bci:Activity','Invalid rates.');
    assert(abs(sum(e(k).target_final)-1)<1e-12,'bci:TargetNormalization','Ideal posterior not normalized.');
    if e(k).likelihood_high(1)==e(k).likelihood_high(2)
        assert(max(abs(e(k).bayes-[.5 .5]))<1e-12,'bci:NoEvidence','Uninformative cue changes posterior.');
    end
end
% Reference independence: switch SPM OFF and reproduce all anchor neural outputs.
without=cfg; without.run_spm_reference=false; without.run_sensitivity=false;
without.grid_levels=.5;
control=run_bayesian_circuit(without);
for k=1:8
    assert(isequal(report.examples(k).raw,control.examples(k).raw), ...
        'bci:ReferenceLeak','Neural trajectories depend on the SPM reference.');
end
report.validation=struct('passed',true,'reference_independence',true, ...
    'raw_tolerance',.08,'normalization_tolerance',.12,'kl_tolerance',.015, ...
    'note','Empirical thresholds for this bounded family, not uniform approximation guarantees.');
fprintf('PASS: Bayes, SPM, fixed hardware, program effects, numeric sensitivity, and reference independence.\n');
end
