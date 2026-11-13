# Continual Program Benchmark v1

A first continual-acquisition extension of Motor Program Benchmark v1. A fixed
attention interpreter acquires uncertain cue–program associations across six
training blocks, archives them, and retrieves programs without a task ID at test.
This is learning within a known six-program language, with known training
boundaries, rather than discovery of new primitives or arbitrary programs.

## Run

In this folder, from MATLAB or Octave:

```matlab
report = run_cpb_benchmark();
```

In the observed Linux environment, inherited Snap library paths interfere with
the Octave launcher. The verified command is:

```sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD octave-cli --quiet --eval "addpath(pwd); report=run_cpb_benchmark();"
python3 make_manifest.py
```

The full run uses the frozen `training_fixture.mat`, validates the execution
kernel, evaluates 60 learning streams, checks learned-context posteriors with
SPM12, and produces numerical exports and editable-source figures. Override
`spm_root` if SPM is not at `~/tools/spm12`; explicitly set
`use_spm_reference=false` to omit that independent check. SPM supplies neither
learning updates nor actions to the prototype. MATLAB compatibility is intended,
but the delivered experiment was run under Octave 11.1.0 only.

A reduced six-stream run, using two motor layouts, is labelled `quick=true`:

```matlab
r=run_cpb_benchmark(struct('quick',true,'output_dir','/tmp/cpb_quick'));
```

To regenerate exports/figures without repeating learning:

```matlab
load('results/cpb_report.mat','report');
cpb_write_results(report,'results');
cpb_plot_results(report,'results');
cpb_plot_acquisition(report,'results');
```

## What changes during learning

The task family contains the six permutations of A/B/C followed by depot D.
Each new training block allocates a memory slot. The environment draws an unknown
assignment of programs and cue identities to tasks. No correct code is supplied
to the learner. It observes 24 noisy context cues and can ask which goal belongs
to program position 1, 2, or 3. Teacher replies are correct with probability .85.
The teacher likelihood is known; the context cue model is learned.

Persistent memory stores a six-component posterior over codes and six Dirichlet
cue counts for each introduced task. Interpreter projection weights never change.
Slots are retained exactly; this is an explicit memory architecture, not an
emergent neural consolidation result. Arrays reserve six slots; logically used
memory is 12 scalars per introduced task, in addition to shared projections,
program grammar, bookkeeping and temporary state. No memory-efficiency claim is
made against a compiled bank or symbolic dictionary.

At test, one noisy cue triggers Bayesian retrieval over introduced slots, then
mixes their code beliefs. The most probable code is executed on the fixed motor
suite. True task identity is used only by the evaluator. Teacher questions occur
during acquisition; this version does not ask a teacher during the test episode.

## Controls

- `active_archive`: one-step cost-minus-information query choice with STOP.
- `random_yoked`: random query positions with exactly the active condition's
  query count for that same stream/block. This matches expenditure, not stopping
  policy; it is a yoked diagnostic.
- `active_fixed` / `random_fixed`: informative versus random queries, both with
  a fixed budget of 12. Acquisition curves retain every query prefix.
- `compiled_matched` / `dictionary_matched`: the active learner and retrieval
  with equivalent specialized or direct GO execution. These are execution
  invariance controls, not independently trained competitors.
- `reset_old_codes` / `reset_old_router`: erase old program posteriors or old
  cue evidence at evaluation, respectively. These are deliberate causal
  interventions, not optimized continual-learning baselines.

See [PROTOCOL.md](PROTOCOL.md) for the objective, information boundary and exact
averaging. The original MPB folder and frozen B1 baseline are unchanged.

## Results and interpretation

The full 60-stream run learned the correct code in **99.17%** of active
acquisitions using **4.19 questions** on average, versus **87.78%** for yoked
random questions. Both 12-question conditions reached **99.72%** code accuracy. At exactly four questions, the fixed-budget prefixes give
91.94% versus 88.61% code accuracy; this isolates a smaller question-allocation
benefit without conflating it with adaptive stopping.
Final inferred-context completion was **69.47%** for the active archive, against
the protocol's **70%** one-cue ceiling. Stored code/count drift was exactly zero.
Compiled and dictionary executors matched the interpreter. Resetting old cue
evidence reduced completion to 27.16%; resetting old code beliefs reduced it to
16.67%. These are finite-fixture measurements, not general CL performance claims.

Eighteen learned-context posterior checks against original SPM12 agreed within
2.23e-15. SPM's finite log convention produces negligible unseen-state mass;
the prototype masks those slots exactly. Core and integration checks include
126 Bayes cases, 42 information audits, 12 retrieval audits, 648 teacher-index
checks, 72 complete trajectory pairs, and 1296 motor rollouts on 36 layouts.

The measured final-stage table is in `results/summary.csv`; acquisition evidence
is in `results/queries.csv` and `results/acquisition.csv`. `retention.csv` contains
every stream/stage/task/condition. Full checkpoints and the exact learning
fixture are in `results/cpb_report.mat`.

`success` is task completion after inferred retrieval. `oracle_context_success`
provides the correct memory slot but still uses the learned code.
`recognition_accuracy` also gives success with a correct code bank because task
programs are distinct in this fixture. `oracle_program_success` checks the motor
executor with the actual program. These diagnostic oracles are never learner
inputs in ordinary testing.

Old archived code/count vectors have zero drift by design and by assertion.
End-to-end performance can nevertheless decline as more contexts compete for an
ambiguous cue. Even a perfect recognizer has mean success 1, .82, .76, .73, .712,
and .70 across stages under this cue protocol. Such decline must not be called
catastrophic forgetting. At the final stage the Bayes cue ceiling is .70.

All sensory outcomes at test are integrated exactly. Acquisition uses 60 frozen
noisy streams and balanced curriculum positions; reported means describe that
finite fixture. No population confidence intervals or general superiority claim
is attached. The two geometry sets are pooled: 24 MPB evaluation and 12 MPB
geometry-transfer layouts, all unused by the teacher-acquisition procedure.

## Sources and next step

- `cpb_config.m`, `cpb_make_fixture.m`: versioned protocol and fixture generation.
- `cpb_select_query.m`, `cpb_code_update.m`, `cpb_train_code.m`: query policy and
  persistent Bayesian code acquisition.
- `cpb_recognize.m`: attention-based context inference and explicit code mixture.
- `cpb_eval_memory.m`: independent outcome integration; hidden labels stay here.
- `test_cpb_core.m`, `test_cpb_integration.m`, `cpb_motor_validation.m`: numerical,
  indexing, and closed-loop checks.
- `cpb_plot_results.m`, `cpb_plot_acquisition.m`: sources for the overview and
  equal-budget acquisition figures.
- `vendor_mpb/`: byte-identical, hashed subset of the preceding MPB implementation.
- [English research document](/media/Data/OneDrive/theory/ActiveAndDeep-theory/Virtuality/virtuality_ctrnn_active_inference.pdf).

This step adds continual acquisition and retrieval. Interactive inference remains
a subsequent experiment: two agents with complementary private cues, joint
subgoals and costly partner-aware communication. Neither a second agent nor
communication has been added silently to this protocol.
