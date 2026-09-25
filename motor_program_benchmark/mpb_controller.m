function probs = mpb_controller(position, goal, p)
%MPB_CONTROLLER Shared, fixed local GO controller; horizontal movement first.
% Coordinates are [row, column]; increasing row means south. No route input.
    if nargin < 3, p = mpb_parameters(); end
    if numel(position) ~= 2 || numel(goal) ~= 2 || ...
            any(~isfinite(position(:))) || any(~isfinite(goal(:))) || ...
            any(position(:) ~= fix(position(:))) || any(goal(:) ~= fix(goal(:)))
        error('mpb:InvalidCoordinates', 'Position and goal need two finite integer coordinates.');
    end
    if position(2) < goal(2), action = 3;
    elseif position(2) > goal(2), action = 4;
    elseif position(1) < goal(1), action = 2;
    elseif position(1) > goal(1), action = 1;
    else, action = 5;
    end
    probs = zeros(1, p.n_actions);
    probs(action) = 1;
end
