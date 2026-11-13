function L = cpb_query_model(phase, c)
%CPB_QUERY_MODEL P(noisy teacher goal report | route code, queried position).
% Rows 1:3 name goals A:C; rows 4:6 are zero-probability padding required
% by the six-state MPB Bayesian-attention interface. The learner receives
% this likelihood, never the hidden teacher route.
    if ~isscalar(phase) || ~isfinite(phase) || phase ~= fix(phase) || phase < 1 || phase > 3
        error('cpb:InvalidPhase', 'Teacher queries must select position 1, 2 or 3.');
    end
    if ~isfield(c, 'routes') || ~isequal(size(c.routes), [6, 4]) || ...
            any(~isfinite(c.routes(:))) || ...
            any(any(sort(c.routes(:, 1:3), 2) ~= repmat(1:3, 6, 1))) || ...
            any(c.routes(:, 4) ~= 4)
        error('cpb:InvalidRoutes', 'Routes must be six permutations of goals 1:3 followed by goal 4.');
    end
    if size(unique(c.routes, 'rows'), 1) ~= 6
        error('cpb:InvalidRoutes', 'All six routes must be distinct.');
    end
    if ~isfield(c, 'demo_reliability') || ~isscalar(c.demo_reliability) || ...
            ~isfinite(c.demo_reliability) || c.demo_reliability < 0 || c.demo_reliability > 1
        error('cpb:InvalidReliability', 'Teacher reliability must lie in [0, 1].');
    end
    L = zeros(6, 6);
    L(1:3, :) = (1 - c.demo_reliability) / 2;
    for code = 1:6
        L(c.routes(code, phase), code) = c.demo_reliability;
    end
end
