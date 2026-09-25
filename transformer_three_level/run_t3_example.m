function report = run_t3_example(cfg)
%RUN_T3_EXAMPLE Three functional levels inside ONE encoder-style head/block.
% report = run_t3_example(struct('save_results', false))
% Optional cfg.X is a 3-token by 3-feature matrix. Trainable parameter
% matrices are hand-chosen illustrative constants; no training is done.
% Bidirectional, unmasked, no positional encoding; an untrained four-symbol
% vocabulary readout follows the block. This is not a language model.
if nargin < 1, cfg = struct(); end
if ~isfield(cfg, 'X'), cfg.X = [1 .2 -.4; -.3 .8 .5; .6 -.5 .9]; end
if ~isfield(cfg, 'save_results'), cfg.save_results = true; end
if ~isfield(cfg, 'output_dir'), cfg.output_dir = fullfile(fileparts(mfilename('fullpath')), 'results'); end
assert(isequal(size(cfg.X), [3 3]) && all(isfinite(cfg.X(:))), ...
    't3:Input', 'X must be a finite 3-by-3 matrix: rows are tokens.');
p = fixed_parameters();
report = forward(cfg.X, p);
report.metadata = struct('name', 'three_functional_levels_one_attention_head', ...
    'runtime', version, 'n_tokens', 3, 'd_model', 3, 'd_k', 3, 'd_v', 3, ...
    'n_heads', 1, 'ffn_hidden_width', 6, 'architecture', 'encoder-style postnorm', ...
    'vocabulary_size', 4, 'readout_trained', false, ...
    'attention_scale', 1/sqrt(3), 'causal_mask', false, 'positional_encoding', false, ...
    'training', false, 'multipliers', 'exact scalar arithmetic; not CTRNN mulation', ...
    'levels', 'Q/K/V projections; attention program construction; value execution', ...
    'scope', 'Three functional stages, not three stacked Transformer blocks.');
report.config = cfg;

% Intervention 1: alter one query only, holding keys and values fixed.
query = report.stage1.Q; query(1, :) = query(1, :) + [.3 -.2 .15];
q = attention_from_qkv(query, report.stage1.K, report.stage1.V);
report.perturbations.query_only = struct('Q', query, 'A', q.A, 'Z', q.Z, ...
    'A_max_change', max(abs(q.A(:)-report.stage2.A(:))), ...
    'Z_max_change', max(abs(q.Z(:)-report.stage3.Z(:))));

% Intervention 2: change another token's context features; keep all learned
% parameters fixed, then recompute its Q/K/V and contextual attention.
context = cfg.X; context(2, :) = context(2, :) + [.3 -.25 .1];
changed = forward(context, p);
report.perturbations.context_token = struct('X', context, ...
    'A', changed.stage2.A, 'Z', changed.stage3.Z, 'Y', changed.downstream.Y, ...
    'A_max_change', max(abs(changed.stage2.A(:)-report.stage2.A(:))), ...
    'Z_max_change', max(abs(changed.stage3.Z(:)-report.stage3.Z(:))), ...
    'Y_max_change', max(abs(changed.downstream.Y(:)-report.downstream.Y(:))));

% Intervention 3: replace the key program while clamping Q and V. This is
% deliberately a controlled coefficient intervention, not a full token swap.
key_program = report.stage1.K([2 3 1], :)';
scores = t3_programmed_linear(report.stage1.Q, key_program, 'right') / sqrt(3);
A = row_softmax(scores);
Z = t3_programmed_linear(report.stage1.V, A, 'left');
report.perturbations.key_program_swap = struct('key_program', key_program, ...
    'A', A, 'Z', Z, 'A_max_change', max(abs(A(:)-report.stage2.A(:))), ...
    'Z_max_change', max(abs(Z(:)-report.stage3.Z(:))));

% Intervention 4: directly replace the row-stochastic attention program,
% while clamping the value array and reusing exactly the same linear loop.
A = [.6 .3 .1; .2 .2 .6; .15 .7 .15];
Z = t3_programmed_linear(report.stage1.V, A, 'left');
report.perturbations.attention_program_swap = struct('A', A, 'Z', Z, ...
    'Z_max_change', max(abs(Z(:)-report.stage3.Z(:))));
report.diagnostics = struct('attention_row_sums', sum(report.stage2.A, 2), ...
    'attention_minimum', min(report.stage2.A(:)), ...
    'ln1_standardized_row_means', mean(report.downstream.ln1.standardized, 2), ...
    'ln2_standardized_row_means', mean(report.downstream.ln2.standardized, 2));
if cfg.save_results, t3_write_results(report, cfg.output_dir); end
end

function p = fixed_parameters()
p.Wq = [.8 -.2 .1; .3 .9 -.4; -.1 .5 .7];
p.Wk = [.6 .4 -.2; -.3 .8 .5; .2 -.1 .9];
p.Wv = [1 .2 -.1; -.2 .7 .4; .3 -.5 .8];
p.Wout = [.7 .1 -.2; -.1 .9 .3; .2 -.3 .8];
p.W1 = [.6 -.4 .2 .8 -.3 .5; -.1 .7 .5 -.2 .4 -.6; .3 .2 -.7 .1 .9 .4];
p.b1 = [.1 -.2 .05 0 .15 -.1];
p.W2 = [.5 -.2 .3; -.4 .6 .1; .2 .3 -.5; .7 .1 -.2; -.3 .4 .6; .1 -.5 .2];
p.b2 = [.05 -.05 .1];
p.gamma1 = [1.1 .9 1]; p.beta1 = [.05 -.03 .02];
p.gamma2 = [.95 1.05 1]; p.beta2 = [0 .02 -.02];
p.layernorm_epsilon = 1e-5;
p.Wreadout = [.7 -.4 .2 .1; -.2 .8 .3 -.5; .4 .1 -.6 .7];
p.breadout = [.05 -.1 .15 0];
end

function r = forward(X, p)
r.inputs = struct('X', X);
r.fixed_parameters = p;
Q = t3_programmed_linear(X, p.Wq, 'right');
K = t3_programmed_linear(X, p.Wk, 'right');
V = t3_programmed_linear(X, p.Wv, 'right');
a = attention_from_qkv(Q, K, V);
r.stage1 = struct('Q', Q, 'K', K, 'V', V);
r.stage2 = struct('key_program', K', 'scores', a.scores, ...
    'scaled_scores', a.scaled_scores, 'A', a.A);
r.stage3 = struct('Z', a.Z);
projected = t3_programmed_linear(a.Z, p.Wout, 'right');
residual1 = X + projected;
[U, ln1] = row_layernorm(residual1, p.gamma1, p.beta1, p.layernorm_epsilon);
hidden = max(0, t3_programmed_linear(U, p.W1, 'right') + repmat(p.b1, 3, 1));
ffn_output = t3_programmed_linear(hidden, p.W2, 'right') + repmat(p.b2, 3, 1);
residual2 = U + ffn_output;
[Y, ln2] = row_layernorm(residual2, p.gamma2, p.beta2, p.layernorm_epsilon);
r.downstream = struct('projected', projected, 'residual1', residual1, ...
    'U', U, 'ffn_hidden', hidden, 'ffn_output', ffn_output, ...
    'residual2', residual2, 'Y', Y, 'ln1', ln1, 'ln2', ln2);
r.readout = t3_vocabulary_readout(Y, p.Wreadout, p.breadout, ...
    {'alpha','beta','gamma','delta'});
end

function a = attention_from_qkv(Q, K, V)
a.scores = t3_programmed_linear(Q, K', 'right');
a.scaled_scores = a.scores / sqrt(size(K, 2));
a.A = row_softmax(a.scaled_scores);
a.Z = t3_programmed_linear(V, a.A, 'left');
end

function probabilities = row_softmax(scores)
probabilities = zeros(size(scores));
for row = 1:size(scores, 1)
    offset = max(scores(row, :)); denominator = 0;
    for column = 1:size(scores, 2)
        probabilities(row, column) = exp(scores(row, column) - offset);
        denominator = denominator + probabilities(row, column);
    end
    for column = 1:size(scores, 2)
        probabilities(row, column) = probabilities(row, column) / denominator;
    end
end
end

function [Y, info] = row_layernorm(X, gamma, beta, epsilon)
Y = zeros(size(X)); standardized = Y;
means = zeros(size(X, 1), 1); variances = means;
for row = 1:size(X, 1)
    total = 0;
    for column = 1:size(X, 2), total = total + X(row, column); end
    means(row) = total / size(X, 2);
    total = 0;
    for column = 1:size(X, 2), total = total + (X(row, column)-means(row))^2; end
    variances(row) = total / size(X, 2);
    for column = 1:size(X, 2)
        standardized(row, column) = (X(row, column)-means(row)) / sqrt(variances(row)+epsilon);
        Y(row, column) = gamma(column)*standardized(row, column) + beta(column);
    end
end
info = struct('row_means', means, 'row_population_variances', variances, ...
    'standardized', standardized, 'epsilon', epsilon, 'gamma', gamma, 'beta', beta);
end
