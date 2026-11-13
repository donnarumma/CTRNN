function c=cpb_config()
%CPB_CONFIG Versioned, bounded first continual-program-acquisition protocol.
c.version='continual_program_benchmark_v1';
c.routes=[sortrows(perms(1:3)),4*ones(6,1)];
c.n_tasks=6; c.n_stages=6; c.n_replications=60;
c.cue_reliability=.70; c.demo_reliability=.85; c.query_cost=.04;
c.max_queries=12; c.cues_per_block=24; c.dirichlet_alpha=1;
c.seed=29092026;
c.conditions={'active_archive','random_yoked','active_fixed','random_fixed', ...
    'compiled_matched','dictionary_matched','reset_old_codes','reset_old_router'};
c.training='Known task boundaries and new memory slot per block; no task ID at test.';
c.scope='Learning cue-program associations in a known finite program language; growing memory.';
c.evaluation='Exact sensor expectations within each of 60 frozen training streams; no population CI.';
end
