function out = mpb_compiled_execute(program, stage, position, landmarks, p)
%MPB_COMPILED_EXECUTE Independent specialization of a fixed motor program.
% No attention head, token array or runtime key generation is used here.
% The finite-beta selector is retained for an exact behavioral comparison.
    if nargin < 5, p = mpb_parameters(); end
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
    % This table is the program-specialized constant in a compiled controller.
    E = exp(-p.motor_beta) * ones(4);
    for s = 1:4, E(s, program(s)) = 1; end
    E = E ./ (1 + 3 * exp(-p.motor_beta));
    primitive = zeros(4, p.n_actions);
    for j = 1:4, primitive(j, :) = mpb_controller(position, landmarks(j, :), p); end
    out.probabilities = E(stage, :) * primitive;
    [~, out.action] = max(out.probabilities);
    out.specialized_selector = E;
end
