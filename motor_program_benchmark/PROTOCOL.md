# Operational protocol v1

`mpb_protocol.m`, the fixed projection definitions, and `layout_manifest.mat` define the executable protocol. Full results were produced on 23 September 2026. No training is performed. Development of the protocol used small smoke runs; this is not a preregistration.

## Inputs and information boundary

The true ordered route is uniformly sampled conceptually from six permutations of A/B/C followed by D. Exact enumeration replaces sampling in evaluation. The agent sees the layout, current position, its own phase memory, cue observations, the known observation models and a library of possible program descriptions. It never receives the true route. Candidate route identities are hypotheses, not hidden-answer inputs.

The top-level driver owns `truth_id` only to generate the cue probabilities and environment feedback and calculate metrics. `mpb_head`, `mpb_belief_update`, `mpb_execute` and `mpb_policy` do not accept it. The integration test verifies identical motor prefixes across hidden truths until acceptance/rejection can legitimately differ.

## Belief and execution passes

Both modes use the same 11×35 token schema and fixed 35×11 query/key matrices and 35×13 value matrix. All three projections read the same X within each pass; X contents differ between recognition and execution. A role mask makes unused rows attend themselves and the designated query attend either the six hypothesis tokens or the four controller tokens. Sparse support masks handle zero-probability hypotheses without introducing infinite key fields.

Recognition uses an observation one-hot plus an intercept as query, log-likelihood and log-prior fields as keys, and one-hot hypothesis values. Therefore the selected output is the exact posterior. Initial and post-LOOK belief updates occur inside this attention calculation. During the subsequently committed route, motor feedback updates accepted progress; continuing latent-rule inference after every motor feedback is outside this v1 protocol. Reported Brier/log loss is explicitly at the route decision.

Execution uses a one-hot phase query with scale beta=20, a runtime permutation/address matrix in the keys, and the four candidate GO command distributions in the values. This is where the higher program configures the generator of a lower execution program. No mask contains the selected goal. The fixed output projection preserves the command distribution, and deterministic first-maximum action selection executes one command. GO populates movement and TOUCH only. WAIT and LOOK are reserved zero-probability slots of the motor readout; the optional initial LOOK is issued directly by the policy evaluator.

## Policy objective

There are six route commitments and one LOOK policy whose six branches choose a route after the new cue. The branch choices minimize the same objective as their parent. Conditional optimization is exact by the mutual-information chain rule and is tested against an independent exhaustive search on a reachable two-outcome problem.

The common future outcome space has 21 entries: cue tag (no LOOK or one of six symbols) × terminal category (success, first touch wrong, second touch wrong). With three distinct required stations, a permutation cannot first disagree only at the third station. Given a layout and policy, the complete deterministic movement/feedback trajectory is determined by this outcome; no sensory information is discarded by this reduction.

`C(cue_tag,category) = exp(-3 × failure) / [7(1 + 2 exp(-3))]` is normalized on this common space. Active expected free energy is expected negative log preference minus mutual information between route and future outcome, plus explicit action costs. An independently evaluated risk-plus-ambiguity expression agrees numerically. Costs are not counted twice in preferences.

The no-epistemic condition removes only the information term. LOOK remains available and may be instrumentally useful. The no-LOOK condition instead restricts the candidate policies. Terminal feedback information is intrinsic to this objective; the episode offers no later action that could use it. The objective therefore need not minimize pragmatic loss.

## Evaluation and controls

Each layout/regime has six equally likely routes, six initial cues, and six possible extra cues. Conditional masses for every true route sum to one, including branches where no LOOK is taken. Every condition uses identical layouts, prior, cue channels, primitive controllers and costs, except the explicitly privileged oracle and the named causal interventions.

The random-sensing control uses the active agent's mean LOOK rate per layout over the equal mixture of sensor regimes and initial cues. It spends the same expected observation cost on the mixture, while ignoring the current cue/regime when allocating observations. All subsequent conditional inference and route choices still use the appropriate model.

The compiled comparator specializes the program's finite-beta attention selector, preserving its small nonzero coefficients. It is mathematically equivalent to the interpreter at the command-distribution level. It shares the policy evaluator and beliefs. It is not a hard-selector approximation. No runtime-speed or neural-efficiency claim follows from the implementation; the compiled helper currently reconstructs its tiny selector table per call.

Frozen motor keys replace only the execution code by the canonical ABCD code. Frozen motor attention uses a fixed A-controller selector. Reset memory forces the query phase back to the first instruction. The environment's actual accepted-progress state is not erased, making the discrepancy observable. These are execution interventions, not legitimate compiled specializations.

All means are finite-suite expectations. `mixture` averages the three regimes equally; each layout and each route are equally weighted. The `reference`/`withheld` program labels refer to code compositions, not training exposure. All candidate hypotheses remain in the known model, and no statistical learned-generalization claim is made. The data permit alternative aggregation without rerunning stochastic episodes.

## Reproducibility and changes

The binary layout fixture is authoritative across MATLAB/Octave, where equal RNG seeds need not imply identical `randperm` output. Its coordinates are also exported as CSV. Source/artifact SHA256 hashes identify the delivered version. Regenerating the fixture or modifying sensor/cost/controller semantics requires an explicitly new protocol version for comparison.

The environment is an open grid, position is fully observed, and actuation is deterministic. The task tests sequencing and information allocation, not obstacle navigation, system identification, learned dynamics or robust continuous motor control. The sparse projection matrices are hand-designed exact arithmetic, not a CTRNN realization. These boundaries are part of the experimental claim.
