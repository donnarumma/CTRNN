function report = test_t3_example(cfg)
%TEST_T3_EXAMPLE Check scalar execution against independent matrix formulas.
if nargin < 1, cfg = struct(); end
cfg.save_results = false;
report = run_t3_example(cfg);
X = report.inputs.X; p = report.fixed_parameters; tolerance = 1e-12;
Q = X*p.Wq; K = X*p.Wk; V = X*p.Wv;
S = Q*K'; scaled = S/sqrt(3);
E = exp(bsxfun(@minus, scaled, max(scaled, [], 2)));
A = bsxfun(@rdivide, E, sum(E, 2)); Z = A*V;
P = Z*p.Wout; R1 = X+P;
U = reference_ln(R1, p.gamma1, p.beta1, p.layernorm_epsilon);
H = max(0, bsxfun(@plus, U*p.W1, p.b1));
F = bsxfun(@plus, H*p.W2, p.b2); R2 = U+F;
Y = reference_ln(R2, p.gamma2, p.beta2, p.layernorm_epsilon);
scalar = {report.stage1.Q,report.stage1.K,report.stage1.V, ...
    report.stage2.scores,report.stage2.scaled_scores,report.stage2.A,report.stage3.Z, ...
    report.downstream.projected,report.downstream.residual1,report.downstream.U, ...
    report.downstream.ffn_hidden,report.downstream.ffn_output,report.downstream.residual2,report.downstream.Y};
vectorized = {Q,K,V,S,scaled,A,Z,P,R1,U,H,F,R2,Y};
names = {'Q','K','V','scores','scaled_scores','A','Z','projected','residual1','U','ffn_hidden','ffn_output','residual2','Y'};
errors = zeros(1,numel(scalar));
for k = 1:numel(scalar)
    assert(isequal(size(scalar{k}),size(vectorized{k})), 't3:Shape','Wrong shape: %s',names{k});
    errors(k)=max(abs(scalar{k}(:)-vectorized{k}(:)));
    assert(errors(k)<tolerance,'t3:Reference','Scalar/vectorized mismatch: %s',names{k});
end
assert(isequal(size(H),[3 6]) && isequal(size(Y),[3 3]),'t3:BlockShape','Wrong FFN/block shape.');
assert(all(A(:)>0) && max(abs(sum(A,2)-1))<tolerance,'t3:Softmax','Invalid attention rows.');
assert(max(abs(report.stage2.scaled_scores(:)-report.stage2.scores(:)/sqrt(3)))<tolerance, ...
    't3:Scale','Attention scaling is missing or incorrect.');
for entry = {'ln1','ln2'}
    info=report.downstream.(entry{1});
    assert(max(abs(mean(info.standardized,2)))<tolerance,'t3:LNMean','LayerNorm must center each token.');
    expected=info.row_population_variances./(info.row_population_variances+p.layernorm_epsilon);
    assert(max(abs(mean(info.standardized.^2,2)-expected))<tolerance, ...
        't3:LNVariance','LayerNorm must use feature-wise population variance and epsilon.');
end

% Changing one query affects only its attention/output row when K,V are held.
q=report.perturbations.query_only;
assert(q.A_max_change>1e-5 && q.Z_max_change>1e-5,'t3:QueryEffect','Query change has no effect.');
assert(max(max(abs(q.A(2:3,:)-A(2:3,:))))<tolerance, ...
    't3:QueryLocality','A clamped-key query change leaked to another row.');
context=report.perturbations.context_token;
assert(context.A_max_change>1e-5 && context.Z_max_change>1e-5, ...
    't3:ContextEffect','Context change has no effect.');
assert(max(abs(context.Z(1,:)-Z(1,:)))>1e-5, ...
    't3:ContextMixing','Changing token 2 must change token 1 through attention.');
key=report.perturbations.key_program_swap;
assert(key.A_max_change>1e-5 && key.Z_max_change>1e-5,'t3:KeyProgram','Key program swap has no effect.');
program=report.perturbations.attention_program_swap;
assert(program.Z_max_change>1e-5,'t3:AttentionProgram','Attention program swap has no effect.');
assert(max(max(abs(program.Z-program.A*V)))<tolerance,'t3:ProgramExecution','Left-program execution mismatch.');

% No positional encoding or mask: permuting tokens must permute outputs.
order=[3 1 2]; permuted_cfg=cfg; permuted_cfg.X=X(order,:);
permuted=run_t3_example(permuted_cfg);
assert(isequal(permuted.fixed_parameters,p),'t3:FixedWeights','Learned parameters changed.');
permutation_error=max(max(abs(permuted.downstream.Y-Y(order,:))));
assert(permutation_error<tolerance,'t3:Equivariance','Unmasked block must be token-permutation equivariant.');
assert(max(max(abs(permuted.stage2.A-A(order,order))))<tolerance, ...
    't3:AttentionEquivariance','Attention must permute both token axes.');

% FFN is shared but applied token by token: changing only U row 2 may not
% alter either of the other output rows of the FFN sublayer.
changedU=U; changedU(2,:)=changedU(2,:)+[.2 -.1 .3];
changedH=max(0,t3_programmed_linear(changedU,p.W1,'right')+repmat(p.b1,3,1));
changedF=t3_programmed_linear(changedH,p.W2,'right')+repmat(p.b2,3,1);
locality_error=max(max(abs(changedF([1 3],:)-F([1 3],:))));
assert(locality_error<tolerance && max(abs(changedF(2,:)-F(2,:)))>1e-5, ...
    't3:FFNLocality','FFN must act independently with shared weights on each token.');
% A separate, untrained vocabulary head follows Y. Its softmax is over
% symbols (four columns), not over the three attention source tokens.
reference_logits=bsxfun(@plus,Y*p.Wreadout,p.breadout);
readout_exp=exp(bsxfun(@minus,reference_logits,max(reference_logits,[],2)));
reference_probabilities=bsxfun(@rdivide,readout_exp,sum(readout_exp,2));
readout_logit_error=max(abs(report.readout.logits(:)-reference_logits(:)));
readout_probability_error=max(abs(report.readout.probabilities(:)-reference_probabilities(:)));
assert(isequal(size(report.readout.logits),[3 4]) && ...
    isequal(size(report.readout.probabilities),[3 4]),'t3:ReadoutShape','Expected 3-by-4 readout.');
assert(readout_logit_error<tolerance && readout_probability_error<tolerance, ...
    't3:ReadoutReference','Readout differs from independent matrix reference.');
assert(all(report.readout.probabilities(:)>0) && ...
    max(abs(sum(report.readout.probabilities,2)-1))<tolerance, ...
    't3:ReadoutNormalization','Invalid vocabulary probabilities.');
[~,expected_indices]=max(reference_probabilities,[],2);
assert(isequal(report.readout.prediction_indices,expected_indices) && ...
    isequal(report.readout.predicted_symbols,reshape(report.readout.vocabulary(expected_indices),[],1)), ...
    't3:ReadoutPrediction','Predictions disagree with vocabulary argmax.');
shifted=t3_vocabulary_readout(report.downstream.Y,p.Wreadout,p.breadout+1000,report.readout.vocabulary);
shift_error=max(abs(shifted.probabilities(:)-report.readout.probabilities(:)));
assert(shift_error<tolerance,'t3:ReadoutShift','Softmax must ignore a common logit shift.');
extreme=t3_vocabulary_readout([1000 0 0;-1000 0 0;0 0 0], ...
    [1 -1 .5 -.5;0 0 0 0;0 0 0 0],zeros(1,4),report.readout.vocabulary);
assert(all(isfinite(extreme.probabilities(:))) && all(extreme.probabilities(:)>=0) && ...
    max(abs(sum(extreme.probabilities,2)-1))<tolerance && ...
    extreme.probabilities(1,1)>.999 && extreme.probabilities(2,2)>.999 && ...
    max(abs(extreme.probabilities(3,:)-.25))<tolerance, ...
    't3:ReadoutStability','Large/tied logits must produce finite normalized probabilities.');
assert(max(max(abs(permuted.readout.probabilities-report.readout.probabilities(order,:))))<tolerance, ...
    't3:ReadoutEquivariance','Shared readout must preserve token permutation.');
report.validation=struct('passed',true,'absolute_tolerance',tolerance, ...
    'reference_names',{names},'reference_max_abs_errors',errors, ...
    'max_scalar_vectorized_error',max(errors),'permutation_max_abs_error',permutation_error, ...
    'ffn_locality_max_abs_error',locality_error,'fixed_weights',true, ...
    'row_softmax',true,'layernorm_feature_axis',true,'query_context_effects',true, ...
    'key_and_attention_program_swaps',true,'permutation_equivariance',true,'ffn_tokenwise',true, ...
    'readout_logit_max_abs_error',readout_logit_error, ...
    'readout_probability_max_abs_error',readout_probability_error, ...
    'readout_shift_max_abs_error',shift_error,'readout_numerical_stability',true);
fprintf('PASS: scalar/vectorized max error %.17g; attention rows positive and normalized.\n',max(errors));
fprintf('PASS: query/context changes and K/A program swaps with fixed learned weights.\n');
fprintf('PASS: postnorm shapes/feature normalization, permutation error %.17g, FFN locality error %.17g.\n', ...
    permutation_error,locality_error);
fprintf('PASS: vocabulary readout logit/probability errors %.17g / %.17g; common-shift error %.17g.\n', ...
    readout_logit_error,readout_probability_error,shift_error);
end

function Y = reference_ln(X,gamma,beta,epsilon)
centered=bsxfun(@minus,X,mean(X,2));
standardized=bsxfun(@rdivide,centered,sqrt(mean(centered.^2,2)+epsilon));
Y=bsxfun(@plus,bsxfun(@times,standardized,gamma),beta);
end
