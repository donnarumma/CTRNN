function p = mpb_parameters()
%MPB_PARAMETERS Fixed, hand-designed shared-input attention interpreter.
% No parameters are learned or changed between programs or episodes.
    p.routes = [sortrows(perms(1:3)), 4 * ones(6, 1)];
    p.actions = {'north', 'south', 'east', 'west', 'touch', 'wait', 'look'};
    p.n_routes = 6;
    p.n_goals = 4;
    p.n_actions = 7;
    p.motor_beta = 20;
    p.d_k = 11;                 % Six observation symbols + intercept + four stages.
    p.d_v = 13;                 % Six hypotheses + seven motor commands.
    p.d_model = 2 * p.d_k + p.d_v;
    p.n_tokens = 11;            % Query, six hypotheses, four motor primitives.
    p.query_token = 1;
    p.hypothesis_tokens = 2:7;
    p.motor_tokens = 8:11;
    p.Wq = [eye(p.d_k); zeros(p.d_k + p.d_v, p.d_k)];
    p.Wk = [zeros(p.d_k); sqrt(p.d_k) * eye(p.d_k); ...
            zeros(p.d_v, p.d_k)];
    p.Wv = [zeros(2 * p.d_k, p.d_v); eye(p.d_v)];
    p.Rbelief = [eye(p.n_routes); zeros(p.n_actions, p.n_routes)];
    p.Rmotor = [zeros(p.n_routes, p.n_actions); eye(p.n_actions)];
end
