function out = cpb_select_query(b, c)
%CPB_SELECT_QUERY One-step epistemic query selection, including STOP.
% G(STOP)=0 and G(query h)=query_cost-I(code;report|h).
% No actual teacher code is accepted. STOP wins ties within 1e-12, then the
% lowest queried position. This is a myopic information objective; it is
% not a planner for subsequent motor reward or a globally optimal stopping
% rule over a multi-query horizon.
    b = b(:)';
    if numel(b) ~= 6 || any(~isfinite(b)) || any(b < 0) || abs(sum(b) - 1) > 1e-10
        error('cpb:InvalidPrior', 'Code belief must be a normalized nonnegative six-vector.');
    end
    if ~isfield(c, 'query_cost') || ~isscalar(c.query_cost) || ...
            ~isfinite(c.query_cost) || c.query_cost < 0
        error('cpb:InvalidQueryCost', 'Query cost must be finite and nonnegative.');
    end
    out.information_gain = zeros(1, 3);
    out.outcome_probs = zeros(3, 3); % rows: queried position; columns: report
    out.expected_posterior_entropy = zeros(1, 3);
    out.expected_posteriors = zeros(3, 3, 6); % position x report x code
    for phase = 1:3
        L = cpb_query_model(phase, c);
        po = L(1:3, :) * b';
        out.outcome_probs(phase, :) = po';
        ig = 0;
        conditional_entropy = 0;
        for observed = 1:3
            if po(observed) == 0, continue; end
            joint = b .* L(observed, :);
            positive = joint > 0;
            posterior = joint / po(observed);
            out.expected_posteriors(phase, observed, :) = reshape(posterior, 1, 1, 6);
            ig = ig + sum(joint(positive) .* log(L(observed, positive) / po(observed)));
            conditional_entropy = conditional_entropy - po(observed) * ...
                sum(posterior(positive) .* log(posterior(positive)));
        end
        if ig < -1e-12
            error('cpb:NegativeInformation', 'Computed mutual information is negative.');
        end
        out.information_gain(phase) = max(0, ig);
        out.expected_posterior_entropy(phase) = conditional_entropy;
    end
    out.G = [0, c.query_cost - out.information_gain];
    best = min(out.G);
    selected = find(out.G <= best + 1e-12, 1, 'first');
    out.phase = selected - 1;
    out.use_query = out.phase > 0;
    positive = b > 0;
    out.prior_entropy = -sum(b(positive) .* log(b(positive)));
    out.information_identity_error = max(abs(out.information_gain - ...
        (out.prior_entropy - out.expected_posterior_entropy)));
end
