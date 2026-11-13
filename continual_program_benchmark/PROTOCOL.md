# Continual Program Benchmark v1 — operational protocol

This is a versioned mechanistic experiment developed after the 2026-09-23 MPB
run. It is not preregistered. The frozen fixture, source hashes and complete
exports identify the delivered run.

## Training and information access

Each of six blocks introduces one new context with a previously unknown program
assignment. Task boundaries and the new memory slot are supplied during training.
All six possible ABC permutations, followed by D, form a known finite hypothesis
language. The hidden mapping is not supplied. Code assignments are bijective in
the fixture, but the learner does not exploit that cross-task constraint.

Cue identities are also randomly assigned; a task emits its main symbol with
probability .70 and each other symbol with probability .06. The learner observes
24 cues in a labelled training block and updates independent Dirichlet counts,
initialized to one per symbol. The teacher channel is distinct and known:
ASK_POSITION(h) reports the true goal at h with probability .85, either other
goal with probability .075. Position h belongs to 1:3. Depot D is part of the
known language, not learned.

The learner's teacher interface returns only the requested noisy report. Random
values are keyed by (stream, block, query index, position), preventing reuse of
one observation as repeated evidence. The program posterior persists after the
block and is not reset between evaluations. The ordinary retrieval API never
accepts a true task index, true program or evaluation label.

## Inference and query selection

Both program-posterior updates and context recognition use the unchanged MPB
common-input attention kernel. It computes normalized prior × likelihood exactly.
For recognition, the likelihood is the Dirichlet posterior predictive cue matrix
and the prior is uniform over introduced slots. The final program belief is the
explicit mixture q(code)=sum_slot q(slot)q(code|slot). Argmax ties select the first
code. No additional learned neural program-mixture stage is claimed.

The adaptive policy compares STOP with three queries:

```text
G(STOP)=0
G(h)=query_cost - I_q(code; teacher_reply | h), query_cost=.04 nats.
```

This is a one-step epistemic acquisition objective, not optimal multistep
stopping and not the original MPB motor-preference objective. It can be obtained
from one-step EFE with uniform preferences over three teacher-report outcomes,
subtracting the common log(3) term. STOP is modelled as an uninformative dummy
outcome on the same alphabet; it has no information gain or cost. Costs use the
same dimensionless units as information. Stop when no query improves the
objective (tolerance 1e-12); otherwise select the lowest-index minimizing query.
There is a hard cap of 12 queries.

The active fixed-budget control selects the maximal-information query at each
step and always spends 12 queries. Its random counterpart uses uniformly
pre-generated positions and the same count. The yoked-random condition uses the
adaptive condition's realized block count but its own noisy query history.
This pairing is expenditure-matched, not a claim of independent equal-stopping
policies. Every query prefix is exported, including the initial uniform prior.

## Evaluation and execution

After each training block, test every introduced context on all six possible
cue symbols, weighted by the true generative probabilities. This is a closed-set,
task-agnostic test over previously introduced tasks; out-of-distribution task
rejection and unsignalled training boundaries are outside this experiment.

For each observation, the selected code depends only on learned counts and code
posteriors. The evaluator then calculates success and cost against each true
context. The environment uses explicit TOUCH feedback and wrong-touch
termination, as in MPB. The code-driven executor sees the selected program,
phase, position and layout, never the true rule.

Motor costs are averaged over 36 frozen layouts: 24 evaluation grids (7×7) and
12 geometry-transfer grids (9×9). A closed-loop interpreter rollout for every
selected/true pair on every layout verifies the analytic GO/dictionary outcome
and cost. Independent complete compiled/interpreted traces on one layout of
each size verify action/state/cost equivalence. Remaining compiled equivalence
is additionally checked at command-distribution level along all interpreter
trajectories. Thus matched executors share learning and chosen codes; their
summary rows are certified equivalence controls, not separately trained runs.

Diagnostics separate inferred-route completion, correct-slot learned-code
completion, context recognition, code-posterior quality, and correct-program
motor execution. Correct-slot/correct-program oracles are diagnostic only.
Erasing old code beliefs and erasing old cue evidence are evaluation-time
interventions on copies; they never corrupt the archived reference learner.

## Retention and scope

After each block, existing code vectors and cue counts are asserted unchanged.
An old-task performance reduction caused by new competing contexts is distinct
from loss of stored information. With perfect context knowledge models but only
one noisy cue, mean recognition across k known contexts is
`.70 + (6-k)*.06/k`; this equals .70 at the final six-context stage. Report this
ceiling alongside empirical performance rather than naming the unavoidable
reduction catastrophic forgetting.

The implementation preallocates six slots. Used persistent sufficient statistics
are six code probabilities plus six cue counts per introduced task. There is no
bounded-memory replacement policy, interpreter plasticity, replay experiment,
new primitive discovery, variable-length syntax, multi-agent interaction, or
learned transfer to a new program language. Those require separately versioned
experiments. A symbolic dictionary with the same inference and primitives is
expected to work as well; this first test isolates acquisition and retrieval,
not a unique performance benefit of neural attention.

## Reproducibility

Sixty frozen streams counterbalance task positions using groups of six cyclic
curricula. Program/cue assignments and training noise vary per stream. Exact
fixture bytes, not cross-platform RNG equivalence, define reproduction. Means
are descriptive of this fixed collection; no independence-based confidence
intervals are attached to curriculum rows or enumerated test outcomes.

The standalone core has independent Bayes/entropy/permutation checks; integration
covers teacher tensor indexing, analytic cue ceilings and closed-loop traces.
The original SPM12 solver is an optional independent learned-context posterior
reference after learning; its source provenance is recorded and it contributes
no learning or action decisions. Do not interpret that check as validation of
the full query policy or continual-learning protocol.
