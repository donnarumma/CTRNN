# Three functional levels in one Transformer-style attention block

This deterministic MATLAB/Octave example uses **three tokens with three features each**, one attention head (`d_model = d_k = d_v = 3`), and a complete encoder-style **postnorm** block. The three levels are functional stages within one attention head, not three stacked Transformer blocks.

1. **Construct Q, K, V:** fixed projection matrices turn the input token features into queries, keys, and values.
2. **Construct the attention program:** `scores = Q*K'`, then `A = softmax_rows(scores/sqrt(3))`. Queries provide the data and the transposed key array provides the changing coefficients to the scalar linear operator. Softmax turns the resulting scores into positive row-normalized attention weights.
3. **Execute the attention program:** `Z = A*V`. The same fixed scalar operation structure combines value vectors using the context-dependent coefficients in `A`.

The implementation makes the data/program interpretation explicit through `t3_programmed_linear(values, coefficients, side)`: a right-side key program acts on `Q`, and a left-side attention program acts on `V`. This is an operational interpretation of the linear operations; it does not mean that standard Transformer hardware rewrites its learned parameter matrices during attention. **The operators and fixed parameters stay fixed; the context-dependent coefficient arrays change.** Both operands remain ordinary numerical arrays.

The loop implementation uses exact scalar multiplication. It does not use CTRNN, SPM, or the approximate historical `mulation` network, and makes no claim of neural emulation or active inference.

## What follows the attention head

The head output `Z` is followed by:

```
projected = Z*Wout
U = LayerNorm_rows(X + projected)
hidden = ReLU(U*W1 + b1)             % 3 -> 6, shared across tokens
ffn_output = hidden*W2 + b2           % 6 -> 3, shared across tokens
Y = LayerNorm_rows(U + ffn_output)
```

The LayerNorm at each stage uses each token's own three features, population variance, epsilon `1e-5`, and fixed per-feature scale/bias parameters. It does not normalize across tokens. The feed-forward sublayer uses the same parameters at every token but does not mix tokens. Attention provides the cross-token mixing.

`Y` is the block output. A separate **untrained illustrative vocabulary head** now follows it:

```matlab
logits = Y*Wreadout + breadout;       % 3 features -> 4 symbols per token
probabilities = softmax_rows(logits);
```

The arbitrary four-symbol vocabulary is `alpha`, `beta`, `gamma`, `delta`. Softmax is computed stably after subtracting each row's largest logit. Its columns index vocabulary symbols, whereas columns of attention `A` index source tokens: these are different probability arrays with different roles. Predictions are the maximum-probability symbol in each token row. The readout parameters are hand-chosen, with no learning; this is **not a trained language model**, and the labels have no learned linguistic meaning.

The example is bidirectional: there is **no causal mask and no positional encoding**. It is consequently token-permutation equivariant, including the shared readout. An autoregressive language model would additionally require a causal architecture, training objective, and trained parameters.

## Fixed numerical choices

All matrices are hand-chosen illustrative constants and are printed in `run_t3_example.m`; there is no training, optimization, randomness, or noise. Rows of `X`, `Q`, `K`, `V`, `Z`, `U`, and `Y` are tokens. Columns are feature coordinates.

```matlab
X  = [ 1   .2  -.4; -.3  .8   .5;  .6 -.5  .9 ];
Wq = [ .8 -.2   .1;  .3  .9  -.4; -.1  .5  .7 ];
Wk = [ .6  .4  -.2; -.3  .8   .5;  .2 -.1  .9 ];
Wv = [ 1   .2  -.1; -.2  .7   .4;  .3 -.5  .8 ];
Wreadout = [.7 -.4 .2 .1; -.2 .8 .3 -.5; .4 .1 -.6 .7];
breadout = [.05 -.1 .15 0];
```

The attention scale is **`1/sqrt(d_k) = 1/sqrt(3)`**, following scaled dot-product attention. If comparing with the unscaled score illustration in the local Figure 2, `scores.csv` is the unscaled `Q*K'`; this example applies softmax to the separate `scaled_scores.csv`. Changing the scale changes attention values.

## Run and verify

From this directory in MATLAB or Octave:

```matlab
report = test_t3_example();
t3_write_results(report, fullfile(pwd, 'results'));
% Optional plotting when the companion plotter and a graphics toolkit exist:
t3_plot_results(report, fullfile(pwd, 'results'));
```

Or run `report = run_t3_example();` to compute and export without running tests. Pass `struct('save_results',false)` to suppress exports, or an explicit `output_dir`. An optional `X` field supplies another finite `3x3` token array; all parameter matrices remain fixed.

The tests compare every intermediate scalar-loop result with an independently written vectorized reference, with tolerance `1e-12`. They also verify attention positivity/row sums, scaling, output shapes, LayerNorm's feature axis and variance convention, cross-token context effects, single-query locality with clamped K/V, changes under controlled K and A program swaps, permutation equivariance, and token-wise FFN locality. Readout tests compare logits and probabilities with independent matrix formulas, verify 3-by-4 shapes and normalized rows, confirm invariance to a common logit shift, and check finite normalized probabilities for extreme or tied logits. The key-program swap clamps Q and V deliberately; it is an intervention on coefficients, not a consistent permutation of an entire self-attention input sequence.

## Exports and source files

- `run_t3_example.m`: fixed constants, forward stages, postnorm block, controlled program/context interventions.
- `t3_vocabulary_readout.m`: exact linear 3-to-4 projection, stable vocabulary softmax, and symbol predictions.
- `t3_programmed_linear.m`: explicit exact scalar linear execution with left/right coefficient inputs.
- `test_t3_example.m`: independent vectorized reference and meaningful structural tests.
- `t3_write_results.m`: full-precision MAT, JSON, and plain numerical CSV exports; English `t3_summary.tex` with separate captioned X/A/Z/Y and final-symbol-probability tables.
- `t3_plot_results.m`: annotated attention matrix and three-dimensional value/output geometry, exported as vector PDF and PNG.

`results/results.json` contains `inputs`, `fixed_parameters`, `stage1`, `stage2`, `stage3`, `downstream`, `readout`, `perturbations`, `diagnostics`, `metadata`, and (after tests) `validation`. Each main matrix and fixed parameter also has its own CSV, without header rows, in matrix row/column order. `results.mat` preserves the complete report. The LaTeX fragment references the equation labels in the accompanying research notes and must be included there to resolve those references.

## Architectural reference

The attention and postnorm block equations follow [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762). Here the hidden width is chosen as 6 for a small illustrative example. Full notes and editable TikZ diagrams are in the sibling theory workspace under `Virtuality/`.

`readout` contains `vocabulary`, `logits`, `probabilities`, `prediction_indices` (one-based), and `predicted_symbols`. The new CSV exports are `readout_logits.csv`, `readout_probabilities.csv`, `readout_prediction_indices.csv`, `Wreadout.csv`, and `breadout.csv`. The preceding block output `Y` is unchanged by the added readout.
