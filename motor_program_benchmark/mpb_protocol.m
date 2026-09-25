function c = mpb_protocol()
%MPB_PROTOCOL Frozen v1 finite evaluation protocol; no training is performed.
c.version='motor_program_benchmark_v1';
c.development_seed=1101; c.evaluation_seed=2201; c.transfer_seed=3301;
c.n_development=6; c.n_evaluation=24; c.n_transfer=12;
c.grid_size=7; c.transfer_grid_size=9;
c.initial_reliability=[.40 .65 .90];
c.regime_names={'ambiguous','intermediate','reliable'};
c.probe_reliability=.85; c.look_cost=.60;
c.failure_penalty=3; c.move_cost=.01; c.touch_cost=.01;
c.cost_sweep=0:.10:1.20;
c.max_steps=160;
c.reference_programs=[1 2 3 4]; c.withheld_programs=[5 6];
c.conditions={'active','no_epistemic','no_look','matched_random_look', ...
    'compiled_matched','oracle','frozen_motor_keys','frozen_motor_attention','reset_memory'};
c.metrics={'success','wrong_goal','moves','touches','motor_steps','looks', ...
    'motor_cost','total_cost','pragmatic_loss','decision_brier','decision_nll','decision_entropy','timeout'};
c.random_LOOK_matching='Per-layout budget matched over uniform mixture of three sensor regimes; random choice ignores current regime and cue.';
c.scope='Exact enumeration of route and cue uncertainty; deterministic motor execution; no training.';
c.selection='Minimum expected free energy; lowest-index tie within 1e-12; one optional LOOK, then committed route.';
c.preferences='C(cue_tag,terminal_category) proportional to exp(-3*failure), on 7 x 3 common outcomes.';
c.feedback='Explicit TOUCH verifies order; wrong TOUCH terminates; walking through landmarks does not verify them.';
c.execution='Shared GO(goal), horizontal then vertical, explicit stage memory; terminal depot is goal 4.';
c.information='IG includes optional probe and terminal failure-depth/success feedback; hidden variable is route.';
c.transfer='Larger grids and unseen layouts; withheld program execution is compositional reuse, not learned generalization.';
end
