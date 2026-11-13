function h = mpb_head(X, mask, p)
%MPB_HEAD One shared-X, fixed-projection scaled dot-product attention head.
% mask contains zero or -Inf; each row must retain at least one key.
    if nargin < 3, p = mpb_parameters(); end
    if ~isequal(size(X), [p.n_tokens, p.d_model]) || any(~isfinite(X(:)))
        error('mpb:InvalidTokens', 'X must be a finite n_tokens-by-d_model matrix.');
    end
    if ~isequal(size(mask), [p.n_tokens, p.n_tokens]) || ...
            any(~(mask(:) == 0 | mask(:) == -Inf)) || ...
            any(all(mask == -Inf, 2))
        error('mpb:InvalidMask', 'Every mask row needs at least one unmasked key.');
    end
    h.X = X;
    h.mask = mask;
    h.Q = X * p.Wq;
    h.K = X * p.Wk;
    h.V = X * p.Wv;
    h.raw_scores = (h.Q * h.K') / sqrt(p.d_k);
    h.scores = h.raw_scores + mask;
    shifted = bsxfun(@minus, h.scores, max(h.scores, [], 2));
    weights = exp(shifted);
    h.A = bsxfun(@rdivide, weights, sum(weights, 2));
    h.Z = h.A * h.V;
end
