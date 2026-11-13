function out = cpb_recognize(cue_counts, q_codes, seen, observation, p)
%CPB_RECOGNIZE Infer a memory slot from a cue, then mix its code beliefs.
% cue_counts(slot,cue) are strictly positive Dirichlet counts. Seen slots
% have a uniform prior. Unseen slots have exactly zero prior. No evaluation
% task identity or hidden code is accepted by this retrieval API.
% q_context is produced by the fixed MPB Bayesian attention head. q_program
% is the explicit matrix mixture q_context*q_codes; this multiplication is
% not claimed to be a second learned neural inference mechanism.
    if nargin < 5, p = mpb_parameters(); end
    if ~isequal(size(cue_counts), [6, 6]) || any(~isfinite(cue_counts(:))) || any(cue_counts(:) <= 0)
        error('cpb:InvalidCueCounts', 'Cue counts must be a strictly positive finite 6-by-6 matrix.');
    end
    seen = seen(:)';
    if numel(seen) ~= 6 || any(~isfinite(seen)) || any(seen ~= 0 & seen ~= 1) || ~any(seen)
        error('cpb:InvalidSeen', 'At least one of six binary memory availability flags must be set.');
    end
    seen = logical(seen);
    if ~isequal(size(q_codes), [6, 6]) || any(~isfinite(q_codes(:))) || any(q_codes(:) < 0) || ...
            any(abs(sum(q_codes(seen, :), 2) - 1) > 1e-10)
        error('cpb:InvalidCodes', 'Seen memory rows must be normalized nonnegative code beliefs.');
    end
    % Unseen rows may be zero-initialized or contain normalized priors; they
    % cannot affect recognition because their memory prior is exactly zero.
    row_sums = sum(q_codes(~seen, :), 2);
    if any(min(abs(row_sums), abs(row_sums - 1)) > 1e-10)
        error('cpb:InvalidCodes', 'Unseen memory rows must be either zero or normalized.');
    end
    L = bsxfun(@rdivide, cue_counts, sum(cue_counts, 2))';
    prior = double(seen) / sum(seen);
    update = mpb_belief_update(prior, L, observation, p);
    out.q_context = update.posterior;
    out.q_program = out.q_context * q_codes;
    [~, out.predicted_code] = max(out.q_program);
    [~, out.predicted_context] = max(out.q_context);
    out.attention = update.attention;
    out.head = update.attention;
    out.update = update;
    out.prior = prior;
    out.likelihood = L;
    reference_context = prior .* L(observation, :);
    reference_context = reference_context / sum(reference_context);
    reference_program = zeros(1, 6);
    for slot = 1:6
        for code = 1:6
            reference_program(code) = reference_program(code) + reference_context(slot) * q_codes(slot, code);
        end
    end
    out.bayes_reference_error = max(abs(out.q_context - reference_context));
    out.mixture_reference_error = max(abs(out.q_program - reference_program));
    out.max_error = max(out.bayes_reference_error, out.mixture_reference_error);
    out.maxerrors = struct('context', out.bayes_reference_error, 'mixture', out.mixture_reference_error);
end
