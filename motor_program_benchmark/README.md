# Motor programs under uncertainty — finite benchmark v1

This proof of concept tests a **shared-input attention interpreter** for motor programs, exact Bayesian cue inference, and active sensing. It is a small mechanistic benchmark for a methods article, not an established benchmark score, a trained Transformer, or a physical CTRNN implementation.

The interpreter uses fixed query/key/value projections. All three projections read the **same token matrix**. Runtime keys configure the operation that produces attention coefficients; those coefficients configure value execution. A fixed motor readout produces a command distribution. The program changes; the projection weights do not.

The English research notes include the methods, equations, result table and figures: [PDF](/media/Data/OneDrive/theory/ActiveAndDeep-theory/Virtuality/virtuality_ctrnn_active_inference.pdf) · [LaTeX source](/media/Data/OneDrive/theory/ActiveAndDeep-theory/Virtuality/motor_program_benchmark.tex).

## Run

From this folder in MATLAB:

```matlab
report = run_mpb_benchmark();
```

Or with Octave:

```bash
octave --quiet --eval "addpath(pwd); report=run_mpb_benchmark();"
python3 make_manifest.py
```

The default run performs the core, policy and environment tests, evaluates the frozen suite, checks 126 six-state posteriors against the installed SPM12 solver, and writes tables, CSV/JSON/MAT data and figures to `results/`.

The SPM source defaults to `~/tools/spm12`; override it with `struct('spm_root','/path/to/spm12')`. It is an independent posterior reference, and supplies no actions or coefficients to the prototype. See [SPM_REFERENCE.md](SPM_REFERENCE.md). If SPM is unavailable, explicitly opt out:

```matlab
report = run_mpb_benchmark(struct('use_spm_reference',false));
```

A small smoke run is explicitly marked nonstandard in its metadata:

```matlab
r = run_mpb_benchmark(struct('quick',true, ...
    'save_results',false,'make_plots',false,'use_spm_reference',false));
```

To regenerate figures without repeating the experiment:

```matlab
load('results/mpb_report.mat','report');
mpb_plot_results(report,'results');
```

Verified execution environment: GNU Octave, with the original installed SPM12 MATLAB source copied byte-for-byte into a temporary dependency directory. This source uses MATLAB-compatible APIs, but a separate MATLAB run has not been performed. Octave's gnuplot toolkit can emit a toolkit warning and a harmless shutdown message; successful assertions and process exit status are the verification criteria.

## Task and frozen protocol

The agent visits A, B and C in one of their six orders, then touches depot D. Movement is deterministic in an open grid. It must issue `TOUCH` to verify each waypoint: merely passing through a landmark has no effect. A wrong touch terminates the episode. A phase register records accepted progress.

A noisy initial cue identifies the hidden route. One optional paid `LOOK` produces a second cue. The agent then **commits to an ordered program**, while a shared `GO(goal)` controller reacts to its current position at every motor step. This v1 does not replan the route during movement. There are no obstacles or learned motor policies.

- 6 development layouts and 24 evaluation layouts on a 7×7 grid; 12 geometry-transfer layouts on a 9×9 grid.
- The initial cue has reliability 0.40, 0.65 or 0.90; the additional cue has reliability 0.85. Regimes are equally weighted and known to the observer.
- `LOOK` costs 0.60; each movement or touch costs 0.01; incorrect completion incurs a preference penalty of 3. The LOOK-cost sweep is 0:0.1:1.2.
- Uniform prior over the six routes. Every route and sensory outcome is evaluated with its exact probability. No Monte Carlo sensory noise is used.
- `layout_manifest.mat` fixes all coordinates across runtimes; the exported `results/layouts.csv` is its readable counterpart. Seeds are retained as provenance, not as a substitute for the frozen coordinates.
- Programs 1–4 are reference compositions and 5–6 are labelled withheld compositions. No weights are trained, and all six hypotheses remain in the known generative model. This tests **compositional execution**, not learned zero-shot generalization.

See [PROTOCOL.md](PROTOCOL.md) and `mpb_protocol.m` for the operational definitions. This is a versioned research protocol, not a preregistered experiment.

## How the five ideas are connected

1. **Bayesian representation.** A row of attention encodes a posterior over six candidate routes. Query/key scores are constructed as log likelihood plus log prior. The distribution is represented across activities; a physical-neuron count is not specified.
2. **Free energy.** Cue recognition computes the categorical variational-free-energy minimum. An explicit model-based evaluator selects between six commitments and an observation-contingent LOOK policy using expected free energy.
3. **Virtuality.** Program tokens describe the order in which reusable GO controllers are called. A fixed interpreter executes different orders. This is a finite instruction family, not a universal network emulator.
4. **Transformer organization.** Both recognition and execution use `Q=X*Wq`, `K=X*Wk`, `V=X*Wv`, `A=softmax(Q*K'/sqrt(dk)+mask)`, `Z=A*V`. The mask selects token roles. The motor mask never selects the correct goal; the program-bearing keys and stage query determine its coefficient.
5. **Readout.** A fixed projection maps motor values to seven commands: north, south, east, west, touch, wait, look. The GO primitives populate movement and touch only; wait and look are reserved zero-probability motor slots. The initial LOOK is issued by the external policy evaluator. The execution readout preserves the command distribution. T3's residual/LayerNorm/FFN block is not used here; this benchmark implements the programmable attention head and a task-specific readout, not a complete trained Transformer block.

The common token matrix has 11 rows and 35 features; key/query width is 11 and value width is 13. Fields support belief and motor roles with the same fixed projections. In the motor pass, beta=20 makes a matching stage/address dominate the other three controllers. The finite soft routing is retained in the independently specialized compiled comparison. Its maximum deviation from the ideal hard controller is about 6.18e-9.

The fixed controller that prepares candidate motor values and the model-based policy evaluator are explicit components outside the attention head. SPM checks beliefs after execution. No claim is made that the complete planner or the exact arithmetic has been realized in CTRNN neurons.

## Controls and results

The evaluation includes active, no-epistemic, no-LOOK, matched random LOOK, compiled, oracle, frozen motor keys, frozen motor attention, and reset-memory conditions. Frozen-key/attention interventions apply only to the motor execution pass, leaving sensory inference and planning intact.

Random LOOK is matched to the active agent's **per-layout budget over the equal mixture of all three regimes**. It ignores the current regime and cue when deciding whether to look, but uses the correct likelihood in inference. Budgets are not matched within each individual-regime panel.

On the 24 evaluation layouts, averaging the three sensor regimes:

| Condition | Exact success | Expected LOOK count | Total action/sensing cost | Pragmatic loss |
| --- | ---: | ---: | ---: | ---: |
| Active interpreter | 86.67% | 0.667 | 0.6101 | 1.0101 |
| No epistemic term | 80.00% | 0.333 | 0.3991 | 0.9991 |
| No LOOK | 65.00% | 0 | 0.1744 | 1.2244 |
| Random LOOK, matched pooled budget | 79.44% | 0.667 | 0.5982 | 1.2149 |
| Compiled, matched information | 86.67% | 0.667 | 0.6101 | 1.0101 |
| Oracle route | 100% | 0 | 0.2321 | 0.2321 |
| Frozen motor keys | 16.67% | 0.667 | 0.4892 | 2.9892 |
| Frozen motor attention | 0% | 0.667 | 0.4550 | 3.4550 |
| Reset phase memory | 0% | 0.667 | 0.4639 | 3.4639 |

Pragmatic loss is `3 × probability of failure + total cost`, excluding the information bonus. Thus the active agent improves success and decision-posterior quality, and beats random sensing at the same observation budget, but **does not dominate the no-epistemic agent's pragmatic loss**. Active and compiled task metrics are identical.

These are exact expectations on a specified finite suite, not estimates from independently sampled episodes; no binomial confidence intervals are attached. The source also exports every layout/route result. Expected feedback information includes terminal success/failure depth, which cannot improve further actions once this episode terminates. The epistemic objective must not be described as a guaranteed instrumental reward advantage.

## Artifacts and source map

- `results/summary.csv`, `layout_metrics.csv`, `route_metrics.csv`: exact aggregate and disaggregated outcomes.
- `results/cost_sweep.csv`: LOOK-cost sensitivity, evaluated on the same layouts.
- `results/belief_checks.csv`: all 126 attention/Bayes comparisons, including before/after free energy.
- `results/mpb_report.mat`: complete numerical report, fixed parameters, traces, test results and SPM checks.
- `results/mpb_benchmark_results.pdf/png`: controls, belief quality and cost sensitivity, produced by `mpb_plot_results.m`.
- `results/mpb_motor_example.pdf/png`: clamped-program execution diagnostics, not sampled active-policy episodes.
- `results/sha256_manifest.json`: source and artifact fingerprints, generated by `make_manifest.py` after all files are finalized.
- `mpb_head.m`, `mpb_parameters.m`: common-input fixed attention mechanism.
- `mpb_belief_update.m`: exact inference inside attention, with support masks for impossible hypotheses.
- `mpb_execute.m`, `mpb_controller.m`: runtime program execution and reusable GO primitive.
- `mpb_compiled_execute.m`: independent finite-beta specialization, not a call back into the interpreter.
- `mpb_policy.m`: exact contingent policy evaluation; no sampled hidden route is an argument.
- `mpb_rollout.m`: environment and explicit acceptance feedback. The sampled hidden route is read only by the environment.
- `mpb_predict_costs.m`: analytic outcome/cost model checked against actual rollouts.
- `test_mpb_core.m`, `test_mpb_policy.m`, `test_mpb_benchmark.m`: numerical, causal and integration checks.

## Positioning for an article

The supported contribution is a transparent mechanistic link between Bayesian inference, two runtime-program handoffs, reusable motor execution and an explicit active-sensing objective. Adjacent platforms include [BabyAI](https://arxiv.org/abs/1810.08272) and [MiniGrid/MiniWorld](https://arxiv.org/abs/2306.13831); this code does not implement or report scores on those benchmarks. [Planning and navigation as active inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC6060791/) is an important control-theoretic precedent.

This v1 does not establish learned representation generalization, superior neural resource efficiency, superiority over contemporary planning/RL methods, or universal CTRNN programmability. Optimized compiled code can share subroutines too. The next substantive extension would be persistent hidden contexts, repeated opportunities to use acquired information, and a broader controller family—not relabelling these finite results as a general benchmark victory.
