# Independent SPM12 posterior reference

`info = mpb_spm_reference(report, cfg)` compares the completed benchmark's six-context belief updates against the installed original SPM12 `spm_MDP_VB_X` solver. It must run **after prototype outputs are computed**. Neither the true route nor the prototype posterior is supplied to SPM: the latter is used only for comparison after SPM inference.

The wrapper reads `report.belief_checks`: `regime`, `initial_cue`, `probe_cue`, normalized positive row `prior`, and row `posterior`. It reconstructs the symmetric six-outcome likelihood from the matching `report.protocol.initial_reliability`, or `probe_reliability` when `probe_cue > 0`. Every likelihood column sums to one. The standard benchmark produces **126 distinct checks**, independent of the number of layouts: three regimes times six initial cues times one initial update plus six possible probe updates.

```matlab
info = mpb_spm_reference(report, struct());
assert(info.nchecks == 126 && info.maxerror < 1e-10);
```

`cfg.spm_root` can provide an expanded absolute pathname. Otherwise the wrapper uses `SPM12_ROOT`, or `fullfile(getenv('HOME'),'tools','spm12')` if unset. Missing source files raise an explicit error; a requested comparison is never silently skipped.

The reference uses a six-state static hidden Markov model, an identity transition, one supplied observation, `T=1`, `tau=1`, and no policies or actions. Its posterior is `X{1}(:,1)`. `tau=1` changes the solver's numerical update, not the probabilistic model. This comparison validates static posterior inference only; it does not validate expected free energy, program selection, motor sequencing, replanning, or a neural CTRNN realization.

Thirteen installed MATLAB source files are copied byte-for-byte into a short-lived temporary directory: `spm_softmax.m`, `spm_dot.m`, `spm_vec.m`, `spm_cross.m`, `spm_zeros.m`, `spm_KL_dir.m`, `spm_betaln.m`, `spm_psi.m`, `spm_unvec.m`, `spm_length.m`, `spm_MDP_VB_X.m`, `spm_MDP_check.m`, and `spm_MDP_G.m`. Every copied file is read back and checked against its source bytes. Only this directory is added to the search path; original sources are never modified. Cleanup restores the previous path and RNG state and removes the cache, including when an error occurs after cache creation.

`info` records per-check SPM and analytical posteriors, errors, likelihoods, priors, observation indices, original solver paths, all dependency paths and byte counts, the solver's source revision marker, and cleanup flags. Cached paths are retained for provenance but are intentionally absent after return. SPM/analytical discrepancies exceeding `1e-10` raise an error; SPM/prototype discrepancies are reported in `maxerror` for the caller to assess.

The locally installed solver identifies itself as `$Id: spm_MDP_VB_X.m 7766 2020-01-05 21:37:39Z karl $`. The installation is not a Git checkout, so no Git commit is asserted. Runtime provenance is read directly from the installed file instead of assuming this revision remains unchanged.

## Verification

The smoke benchmark's complete set of 126 belief updates was checked in Octave. The maximum SPM/prototype difference was `9.992007221626409e-16`; the maximum SPM/analytical difference was `7.771561172376096e-16`. All thirteen source copies were byte-identical, and path restoration, RNG restoration, and cache removal passed.

A negative control changed only the supplied prototype posterior to the uniform distribution. SPM still matched analytical Bayes (`2.7755575615628914e-16`) and reported the deliberate prototype disagreement (`0.23333333333333314`), confirming that the supplied posterior cannot drive the reference. A nonexistent requested SPM root raised `mpb:SPMMissing` as intended. The check count is independent of the development/evaluation/transfer layout counts.
