# Transformer-style program execution with epistemic active inference

This deterministic proof of concept combines an active probe choice, an attention calculation of Bayesian beliefs, and a fixed executor whose runtime coefficients are those beliefs. Every result is obtained by exact finite enumeration. There is no training, Monte Carlo sampling, or approximate CTRNN multiplication.

The intended directory is `~/tools/CTRNN/transformer_virtuality_active_inference/`. The earlier uncoupled B1 configuration and the separate Transformer block example are unchanged.

## Run

In MATLAB or Octave, from this directory:

```matlab
report = test_tva_rule_task(struct('save_results', true));
```

For ordinary execution without the additional assertions:

```matlab
report = run_tva_rule_task;
```

With the optional independent SPM reference installed alongside these files:

```matlab
report = test_tva_rule_task(struct( ...
    'use_spm_reference', true, ...
    'spm_root', fullfile(getenv('HOME'), 'tools', 'spm12'), ...
    'save_results', true));
```

The optional helper `tva_spm_reference.m` uses the original SPM solver only after all primary outputs have been computed. Missing requested SPM support raises an error. Its comparison covers 72 combinations: two modes, three actions, three outcomes, and four positive priors (uniform and three near-point priors). SPM does not supply programs or select probes.

## Task and the three stages

The hidden rule `c` is uniformly distributed over three row selectors. A payload `D` contains three one-hot class symbols, in an unknown-to-the-selector but fully visible order. All six permutations are evaluated. The correct class is the symbol in row `c`; the payload itself carries no evidence about `c`.

There are three possible sensor outcomes and three actions: `noop`, `left_probe`, and `right_probe`. The noop likelihood is uniform and independent of `c`. For a sensor of reliability `r`, its likelihood matrix is `r*eye(3) + (1-r)/2*(ones(3)-eye(3))`, with observation rows and hidden-rule columns. In known mode 1, left/right reliabilities are `0.9/0.55`; in known mode 2 they are swapped. The mode tells the agent which sensor is reliable, not the hidden rule. Costs are `[0, 0.12, 0.12]` nats. Every episode permits one observation opportunity.

1. **Generate queries, keys and values with fixed matrices.** For each candidate rule, the key input is `[log(prior_c), log(L_a(o,c)), 0]`. Fixed projections produce `Q=[1,0,0]` and `K_c=[sqrt(3)*(log(prior_c)+log(L_a(o,c))),0,0]`. Values are `V=D`.
2. **Infer a runtime program.** `q=softmax(Q*K'/sqrt(3))` is exactly the posterior over the three rules. The key projection compensates the attention scaling; dividing unadjusted log probabilities by `sqrt(3)` would instead temper the posterior.
3. **Execute the program.** The same scalar multiply-accumulate routine computes `z=q*D`. For the one-hot payloads, this is the exact posterior-predictive class distribution.

Three features are a chosen representation, not a count of three physical CTRNN neurons. The first-stage log probabilities are a model/evidence encoding, not learned logarithms inside a biological network. This example establishes an executable decomposition within a finite family of row-selection programs; it does not establish universal neural emulation.

## Active inference and the ablation

For prior `pi` and action `a`, the information gain is

```text
I(a) = sum_c,o pi(c) L_a(o,c) log[L_a(o,c) / sum_d pi(d)L_a(o,d)]
G(a) = cost(a) - I(a)
```

This has an explicit expected-free-energy derivation. Let the outcome be `(o,s)`, where `s` reports the chosen sensing action deterministically. Set `C(o,s)=exp(-cost(s))/(3*sum_b exp(-cost(b)))`. Joint-outcome risk plus ambiguity equals `G(a) + log(3) + log(sum_b exp(-cost(b)))`. The extra term is common to all actions. The code verifies this identity directly. Outcome preferences are uniform apart from sensing cost; classification accuracy is evaluated afterward and is not secretly included as a reward in this policy objective.

The minimum objective selects the action. Exactly tied actions receive equal probabilities and are enumerated; there is no random draw. This is a one-step epistemic special case of active inference, not a long-horizon planner.

## What follows the attention output

The decision readout follows the same postnorm pattern as the separate Transformer example:

```text
U = LN(ones(1,3) + z*I)
FFN(U) = ReLU(U*[I,-I]) * (0.2*[I; -0.5*I])
Y = LN(U + FFN(U))
pclass = softmax(2*Y)
```

Layer normalization uses the population variance across each row's three features, epsilon `1e-5`, unit scales and zero shifts. The FFN is nonzero and shared across rows. The symmetric monotone construction preserves maximizing classes, but changes the probability values.

**`pclass` is a decision readout, not the exact Bayesian predictive distribution `z`.** The exports report its log loss separately from Bayesian-predictive log loss. Higher classification accuracy need not imply better decision-head log loss: the untrained head can be overconfident, especially with the weaker probe. No vocabulary prediction or trained class head is claimed.

## Controls and exact evaluation

| Condition | Sensing | Execution |
|---|---|---|
| `active` | Minimize cost minus information gain | Posterior program |
| `no_epistemic` | Same options and costs, remove information gain only | Posterior program |
| `passive` | Force noop | Posterior program |
| `random_probe` | Left/right with probability one half each | Posterior program |
| `fixed_program`, `_2`, `_3` | Same probe as active | Freeze the executor at selector 1, 2, or 3 |
| `oracle` | Same EFE with a privileged near-point prior, epsilon `1e-9` | Posterior program |
| `compiled_matched` | Same policy and posterior as active | Three dedicated selection branches followed by the same weighted mixture |

All three fixed selectors are equally good under the uniform prior and balanced payload permutations. They are reported individually. These controls still calculate the sensor posterior for diagnostics, but override the program actually executed. Consequently `posterior_true` and `program_true` differ in these rows. Their Bayesian-predictive NLL is the diagnostic inference reference before the override; actual performance and decision-head NLL follow the frozen execution.

Active and random probing both purchase exactly one observation at cost `0.12`; their expected accuracies are `0.9` and `0.725`. The no-epistemic, passive, and fixed-program accuracies are `1/3`. Oracle accuracy is one. The matched compiled bank gives the same outputs and accuracy as active interpreted execution; no advantage over that equivalent computation is asserted.

The complete probability mass of each condition is one. Correctness on tied output maxima is `1/number_of_tied_classes` when the target belongs to the tied set, avoiding arbitrary floating-point or index tie breaking. Costs, log losses, entropy, mean true-rule posterior, mean true-program coefficient, and execution MSE are retained. `cost_sweep.csv` exposes the switch from informative probing to noop when cost exceeds available information gain. At zero cost, the ablated objective is tied across all three actions and averages them explicitly.

## Public APIs and outputs

- `tva_model(cfg)`: positive sensor likelihoods, mode labels, prior and six payload permutations.
- `tva_policy(prior,L,costs,condition)`: predicted outcomes, information gain, EFE terms and action probabilities. It has no hidden-rule or payload argument.
- `tva_actor(prior,L,action,observation,payload,parameters,selector_override)`: posterior, program, three stages, compiled reference and downstream readout. It has no hidden-rule or target-class argument.
- `tva_readout(z,parameters)`: shared postnorm/FFN/class computation; supports multiple independent rows.
- `tva_parameters()`: all fixed matrices.
- `tva_write_results(report,output_dir)`: exporter; can re-export a saved report without rerunning the experiment.

Exports: `results.mat`, `results.json`, `summary.csv`, `trials.csv`, `cost_sweep.csv`, and the English `tva_summary.tex`. The latter links to the equations in the research notes. The report includes `posterior_checks` for the independent SPM reference; if enabled, its results are under `report.spm`.

Tests check independent analytic Bayes, dense versus scalar arithmetic, compiled equivalence, payload-independent beliefs, rule relabeling, fixed parameters, direct program interventions, the joint-preference EFE identity, reliability-mode reversal, genuine epistemic ablation, information/cost thresholds, uninformative and nearly certain cases, probability mass, matched sensing cost, numerical ties, the nonzero FFN, and row-independent readout. The actor interface and pre-observation policies are checked to prevent hidden-rule information from entering the agent, except in the explicitly labeled oracle control.

## Figures and sources

After running the tests and exporting the report, use `tva_plot_results(report, output_dir)` to produce `tva_task_results.pdf` and `.png`. The plot compares controls and shows explicitly evaluated sensing costs. Editable architectural diagrams and English equations are in the theory workspace, `Virtuality/transformer_virtuality_active_inference.tex`. The separate [`SPM_REFERENCE.md`](SPM_REFERENCE.md) documents the independent numerical comparison.
