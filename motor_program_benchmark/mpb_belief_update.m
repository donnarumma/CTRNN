function out = mpb_belief_update(prior, L, observation, p)
%MPB_BELIEF_UPDATE Exact categorical Bayesian inference by shared-X attention.
% L(observation, route), with each column a normalized outcome distribution.
% True route/context is deliberately absent from this function's inputs.
    if nargin < 4, p = mpb_parameters(); end
    prior = prior(:)';
    n = p.n_routes;
    if numel(prior) ~= n || any(~isfinite(prior)) || any(prior < 0) || ...
            abs(sum(prior) - 1) > 1e-10
        error('mpb:InvalidPrior', 'Prior must be a normalized nonnegative distribution.');
    end
    if ~isequal(size(L), [n, n]) || any(~isfinite(L(:))) || any(L(:) < 0) || ...
            any(abs(sum(L, 1) - 1) > 1e-10)
        error('mpb:InvalidLikelihood', 'Likelihood columns must be normalized distributions.');
    end
    if ~isscalar(observation) || ~isfinite(observation) || ...
            observation ~= fix(observation) || observation < 1 || observation > n
        error('mpb:InvalidObservation', 'Observation must index a likelihood row.');
    end
    support = prior > 0 & L(observation, :) > 0;
    if ~any(support)
        error('mpb:ImpossibleObservation', 'Observation has zero probability under the prior.');
    end
    % Finite placeholders ensure that masked zeros never cause 0 * (-Inf).
    logL = zeros(size(L));
    logL(L > 0) = log(L(L > 0));
    logprior = zeros(size(prior));
    logprior(prior > 0) = log(prior(prior > 0));
    X = zeros(p.n_tokens, p.d_model);
    X(1, observation) = 1;
    X(1, n + 1) = 1;
    for j = 1:n
        row = p.hypothesis_tokens(j);
        X(row, p.d_k + (1:n + 1)) = [logL(:, j)', logprior(j)];
        X(row, 2 * p.d_k + j) = 1;
    end
    mask = -Inf(p.n_tokens);
    mask(1:p.n_tokens + 1:end) = 0;
    mask(1, :) = -Inf;
    mask(1, p.hypothesis_tokens(support)) = 0;
    h = mpb_head(X, mask, p);
    posterior = h.Z(1, :) * p.Rbelief;
    joint = prior .* L(observation, :);
    evidence = sum(joint);
    logjoint = logprior(support) + logL(observation, support);
    shift = max(logjoint);
    logevidence = shift + log(sum(exp(logjoint - shift)));
    reference = zeros(1, n);
    reference(support) = exp(logjoint - logevidence);
    out.posterior = posterior;
    out.attention = h;
    out.prior = prior;
    out.observation = observation;
    out.evidence = evidence;
    out.log_evidence = logevidence;
    out.bayes_reference = reference;
    out.bayes_reference_error = max(abs(posterior - reference));
    out.F_at_prior = free_energy(prior, prior, L(observation, :));
    out.F_at_posterior = free_energy(posterior, prior, L(observation, :));
    positive = posterior > 0;
    out.KL_posterior_prior = sum(posterior(positive) .* ...
        (log(posterior(positive)) - log(prior(positive))));
    out.F_optimum = -logevidence;
    out.free_energy_identity_error = abs(out.F_at_posterior - out.F_optimum);
    out.posterior_entropy = -sum(posterior(positive) .* log(posterior(positive)));
end

function F = free_energy(r, prior, likelihood)
    positive = r > 0;
    if any(prior(positive) == 0) || any(likelihood(positive) == 0)
        F = Inf;
    else
        F = sum(r(positive) .* (log(r(positive)) - ...
            log(prior(positive)) - log(likelihood(positive))));
    end
end
