function out = mpb_execute(program, stage, position, landmarks, intervention, p)
%MPB_EXECUTE Program-conditioned attention over four shared motor primitives.
% Attention masks encode roles only; goal selection is computed by Q * K'.
    if nargin < 6, p = mpb_parameters(); end
    if nargin < 5 || isempty(intervention), intervention = 'none'; end
    program = program(:)';
    if numel(program) ~= 4 || ~isequal(sort(program(1:3)), 1:3) || program(4) ~= 4
        error('mpb:InvalidProgram', 'Program must permute goals 1:3, then terminate at goal 4.');
    end
    if ~isscalar(stage) || ~isfinite(stage) || stage ~= fix(stage) || stage < 1 || stage > 4
        error('mpb:InvalidStage', 'Stage must be an integer between 1 and 4.');
    end
    if ~isequal(size(landmarks), [4, 2]) || any(~isfinite(landmarks(:)))
        error('mpb:InvalidLandmarks', 'Landmarks must have four finite coordinate rows.');
    end
    valid = {'none', 'frozen_keys', 'frozen_attention', 'reset_memory'};
    if ~ischar(intervention) || ~any(strcmp(intervention, valid))
        error('mpb:InvalidIntervention', 'Unknown execution intervention.');
    end
    requested = zeros(4);
    for s = 1:4, requested(s, program(s)) = 1; end
    effective = requested;
    if strcmp(intervention, 'frozen_keys'), effective = eye(4); end
    query_stage = stage;
    if strcmp(intervention, 'reset_memory'), query_stage = 1; end
    X = zeros(p.n_tokens, p.d_model);
    X(1, p.n_routes + 1 + query_stage) = p.motor_beta;
    controllers = zeros(4, p.n_actions);
    for j = 1:4
        row = p.motor_tokens(j);
        X(row, p.d_k + (8:11)) = effective(:, j)';
        controllers(j, :) = mpb_controller(position, landmarks(j, :), p);
        X(row, 2 * p.d_k + p.n_routes + (1:p.n_actions)) = controllers(j, :);
    end
    mask = -Inf(p.n_tokens);
    mask(1:p.n_tokens + 1:end) = 0;
    mask(1, :) = -Inf;
    mask(1, p.motor_tokens) = 0;
    h = mpb_head(X, mask, p);
    if strcmp(intervention, 'frozen_attention')
        h.nominal_A = h.A;
        h.A(1, :) = 0;
        h.A(1, p.motor_tokens(1)) = 1;
        h.Z = h.A * h.V;
    end
    h.intervention = intervention;
    out.probabilities = h.Z(1, :) * p.Rmotor;
    [~, out.action] = max(out.probabilities);
    out.attention = h;
    out.program = program;
    out.stage = stage;
    out.query_stage = query_stage;
    out.program_matrix = effective;
    out.requested_program_matrix = requested;
    out.primitive_probabilities = controllers;
    out.ideal_probabilities = controllers(program(stage), :);
    out.intervention = intervention;
end
