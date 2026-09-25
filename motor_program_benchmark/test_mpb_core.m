function report = test_mpb_core()
%TEST_MPB_CORE Numerical and causal checks for the fixed interpreter core.
    p = mpb_parameters();
    original_p = p;
    assert(isequal(p.routes, [sortrows(perms(1:3)), 4 * ones(6, 1)]));
    assert(isequal(size(p.Wq), [35, 11]));
    assert(isequal(size(p.Wk), [35, 11]));
    assert(isequal(size(p.Wv), [35, 13]));
    assert(isequal(p.Wq(1:11, :), eye(11)));
    assert(isequal(p.Wv(23:35, :), eye(13)));
    assert(all(sum(p.Wq ~= 0, 1) == 1));
    assert(all(sum(p.Wk ~= 0, 1) == 1));
    assert(all(sum(p.Wv ~= 0, 1) == 1));
    assert(isequal(p.Rbelief, [eye(6); zeros(7, 6)]));
    assert(isequal(p.Rmotor, [zeros(6, 7); eye(7)]));

    priors = [ones(1, 6) / 6; [1, 2, 3, 4, 5, 6] / 21; ...
              1, 0, 0, 0, 0, 0; .2, 0, .3, 0, .5, 0];
    dense = .78 * eye(6) + .22 / 5 * (ones(6) - eye(6));
    sparseL = zeros(6);
    for j = 1:6
        sparseL(j, j) = .7;
        sparseL(mod(j, 6) + 1, j) = .3;
    end
    likelihoods = {dense, ones(6) / 6, eye(6), sparseL};
    report.belief_cases = 0;
    report.impossible_observation_cases = 0;
    report.max_bayes_error = 0;
    report.max_free_energy_error = 0;
    report.max_scalar_head_error = 0;
    for lp = 1:numel(likelihoods)
        L = likelihoods{lp};
        for pr = 1:size(priors, 1)
            prior = priors(pr, :);
            for observation = 1:6
                evidence = sum(prior .* L(observation, :));
                if evidence == 0
                    expect_error(@() mpb_belief_update(prior, L, observation, p), ...
                                 'mpb:ImpossibleObservation');
                    report.impossible_observation_cases = report.impossible_observation_cases + 1;
                    continue;
                end
                b = mpb_belief_update(prior, L, observation, p);
                reference = prior .* L(observation, :) / evidence;
                err = max(abs(b.posterior - reference));
                assert(err < 1e-12);
                assert(abs(sum(b.posterior) - 1) < 1e-12);
                assert(all(b.posterior >= 0));
                assert(all(isfinite(b.attention.X(:))));
                assert(b.free_energy_identity_error < 1e-12);
                assert(abs(b.F_at_posterior + log(evidence)) < 1e-12);
                assert(b.F_at_posterior <= b.F_at_prior + 1e-12);
                assert(isequal(b.attention.Q, b.attention.X * p.Wq));
                assert(isequal(b.attention.K, b.attention.X * p.Wk));
                assert(isequal(b.attention.V, b.attention.X * p.Wv));
                report.belief_cases = report.belief_cases + 1;
                report.max_bayes_error = max(report.max_bayes_error, err);
                report.max_free_energy_error = max(report.max_free_energy_error, ...
                    b.free_energy_identity_error);
            end
        end
    end
    b = mpb_belief_update(priors(2, :), dense, 4, p);
    report.max_scalar_head_error = scalar_head_check(b.attention, p);
    sparse_b = mpb_belief_update(priors(1, :), eye(6), 2, p);
    assert(isinf(sparse_b.F_at_prior));
    assert(isequal(sparse_b.posterior, [0, 1, 0, 0, 0, 0]));
    expect_error(@() mpb_belief_update(ones(1, 6), dense, 1, p), 'mpb:InvalidPrior');
    expect_error(@() mpb_belief_update(priors(1, :), zeros(6), 1, p), 'mpb:InvalidLikelihood');
    expect_error(@() mpb_belief_update(priors(1, :), dense, 7, p), 'mpb:InvalidObservation');

    landmarks = [1, 3; 3, 5; 5, 3; 3, 1];
    report.execution_cases = 0;
    report.max_compiled_error = 0;
    report.max_finite_beta_error = 0;
    for route = 1:6
        program = p.routes(route, :);
        for stage = 1:4
            for row = 1:5
                for col = 1:5
                    position = [row, col];
                    m = mpb_execute(program, stage, position, landmarks, 'none', p);
                    c = mpb_compiled_execute(program, stage, position, landmarks, p);
                    err = max(abs(m.probabilities - c.probabilities));
                    assert(err < 1e-12);
                    assert(m.action == c.action);
                    [~, ideal_action] = max(m.ideal_probabilities);
                    assert(m.action == ideal_action);
                    assert(abs(sum(m.probabilities) - 1) < 1e-12);
                    assert(all(m.attention.mask(1, p.motor_tokens) == 0));
                    assert(all(m.attention.mask(1, 1:7) == -Inf));
                    assert(isequal(m.attention.Q, m.attention.X * p.Wq));
                    assert(isequal(m.attention.K, m.attention.X * p.Wk));
                    assert(isequal(m.attention.V, m.attention.X * p.Wv));
                    report.execution_cases = report.execution_cases + 1;
                    report.max_compiled_error = max(report.max_compiled_error, err);
                    report.max_finite_beta_error = max(report.max_finite_beta_error, ...
                        max(abs(m.probabilities - m.ideal_probabilities)));
                end
            end
        end
    end
    m = mpb_execute([3, 2, 1, 4], 2, [3, 3], landmarks, 'none', p);
    report.max_scalar_head_error = max(report.max_scalar_head_error, scalar_head_check(m.attention, p));
    % Changing only the runtime program changes K and A, but not Q or V.
    a = mpb_execute([1, 2, 3, 4], 1, [3, 3], landmarks, 'none', p);
    b = mpb_execute([3, 2, 1, 4], 1, [3, 3], landmarks, 'none', p);
    assert(isequal(a.attention.Q, b.attention.Q));
    assert(isequal(a.attention.V, b.attention.V));
    assert(isequal(a.attention.mask, b.attention.mask));
    assert(~isequal(a.attention.K, b.attention.K));
    assert(a.action == 1 && b.action == 2); % North vs south at the same position.
    frozen = mpb_execute([3, 2, 1, 4], 1, [3, 3], landmarks, 'frozen_keys', p);
    assert(isequal(frozen.attention.K, a.attention.K));
    assert(frozen.action == a.action);
    assert(isequal(frozen.requested_program_matrix, b.program_matrix));
    frozen_a = mpb_execute([3, 2, 1, 4], 2, [3, 3], landmarks, 'frozen_attention', p);
    assert(isequal(frozen_a.attention.A(1, p.motor_tokens), [1, 0, 0, 0]));
    assert(frozen_a.action == 1);
    assert(isfield(frozen_a.attention, 'nominal_A'));
    reset = mpb_execute([3, 2, 1, 4], 2, [3, 3], landmarks, 'reset_memory', p);
    assert(reset.stage == 2 && reset.query_stage == 1 && reset.action == 2);
    expect_error(@() mpb_execute([1, 1, 3, 4], 1, [3, 3], landmarks, 'none', p), ...
                 'mpb:InvalidProgram');
    expect_error(@() mpb_execute([1, 2, 3, 4], 5, [3, 3], landmarks, 'none', p), ...
                 'mpb:InvalidStage');
    invalid_mask = -Inf(p.n_tokens);
    expect_error(@() mpb_head(zeros(p.n_tokens, p.d_model), invalid_mask, p), 'mpb:InvalidMask');
    assert(isequal(p, original_p));
    report.parameter_immutability = true;
    report.program_intervention_checks = true;
    report.passed = true;
    fprintf(['MPB core PASS: %d feasible belief cases, %d impossible observations, ' ...
             '%d execution cases. Bayes %.3g; free energy %.3g; compiled %.3g; scalar %.3g.\n'], ...
        report.belief_cases, report.impossible_observation_cases, report.execution_cases, ...
        report.max_bayes_error, report.max_free_energy_error, report.max_compiled_error, ...
        report.max_scalar_head_error);
end

function maxerr = scalar_head_check(h, p)
% Independently evaluate every product and accumulation, including projections.
    n = p.n_tokens;
    Q = zeros(n, p.d_k); K = Q; V = zeros(n, p.d_v);
    for i = 1:n
        for j = 1:p.d_k
            for r = 1:p.d_model
                Q(i, j) = Q(i, j) + h.X(i, r) * p.Wq(r, j);
                K(i, j) = K(i, j) + h.X(i, r) * p.Wk(r, j);
            end
        end
        for j = 1:p.d_v
            for r = 1:p.d_model, V(i, j) = V(i, j) + h.X(i, r) * p.Wv(r, j); end
        end
    end
    S = h.mask;
    for i = 1:n
        for j = 1:n
            score = 0;
            for r = 1:p.d_k, score = score + Q(i, r) * K(j, r); end
            S(i, j) = S(i, j) + score / sqrt(p.d_k);
        end
    end
    A = zeros(n);
    for i = 1:n
        maxscore = max(S(i, :));
        denom = 0;
        for j = 1:n, denom = denom + exp(S(i, j) - maxscore); end
        for j = 1:n, A(i, j) = exp(S(i, j) - maxscore) / denom; end
    end
    Z = zeros(n, p.d_v);
    for i = 1:n
        for j = 1:p.d_v
            for r = 1:n, Z(i, j) = Z(i, j) + A(i, r) * V(r, j); end
        end
    end
    errors = [max(abs(Q(:) - h.Q(:))), max(abs(K(:) - h.K(:))), ...
              max(abs(V(:) - h.V(:))), max(abs(A(:) - h.A(:))), max(abs(Z(:) - h.Z(:)))];
    maxerr = max(errors);
    assert(maxerr < 1e-12);
end

function expect_error(fn, expected_id)
    got_expected = false;
    try
        fn();
    catch err
        got_expected = strcmp(err.identifier, expected_id);
        if ~got_expected, rethrow(err); end
    end
    assert(got_expected, ['Expected exception: ', expected_id]);
end
