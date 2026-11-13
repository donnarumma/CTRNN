function stats = test_cpb_core()
%TEST_CPB_CORE Deterministic mathematical and information-boundary tests.
% Independent direct Bayes and entropy calculations audit the attention
% implementation. Hidden truth appears only in synthetic teacher fixtures.
    c.routes = [sortrows(perms(1:3)), 4 * ones(6, 1)];
    c.demo_reliability = .85;
    c.query_cost = .04;
    c.max_queries = 12;
    p = mpb_parameters();
    original_p = p;
    stats = struct('belief_checks', 0, 'query_checks', 0, 'recognition_checks', 0, ...
        'max_bayes_error', 0, 'max_free_energy_error', 0, ...
        'max_information_error', 0, 'max_recognition_error', 0, ...
        'max_permutation_error', 0, 'learned_programs', 0);
    beliefs = [ones(1, 6) / 6; [1 2 3 4 5 6] / 21; ...
        [.2 0 .3 0 0 .5]; [0 0 0 0 1 0]];
    for k = 1:10
        row = 1 + mod((1:6) .* (k + 1) + k^2, 17);
        beliefs(end + 1, :) = row / sum(row); %#ok<AGROW>
    end
    for ib = 1:size(beliefs, 1)
        b = beliefs(ib, :);
        query = cpb_select_query(b, c);
        assert(query.phase >= 0 && query.phase <= 3);
        H = entropy_of(b);
        for phase = 1:3
            L = cpb_query_model(phase, c);
            assert(isequal(size(L), [6, 6]));
            assert(max(abs(sum(L, 1) - 1)) < 1e-14);
            assert(all(all(L(4:6, :) == 0)));
            expected_H = 0;
            expected_b = zeros(1, 6);
            for observed = 1:3
                u = cpb_code_update(b, phase, observed, c, p);
                joint = b .* L(observed, :);
                po = sum(joint);
                direct = joint / po;
                err = max(abs(u.posterior - direct));
                stats.max_bayes_error = max(stats.max_bayes_error, err);
                stats.max_free_energy_error = max(stats.max_free_energy_error, ...
                    abs(u.F_at_posterior + log(po)));
                assert(err < 1e-12);
                assert(u.F_at_posterior <= u.F_at_prior + 1e-12);
                assert(u.free_energy_identity_error < 1e-12);
                expected_H = expected_H + po * entropy_of(u.posterior);
                expected_b = expected_b + po * u.posterior;
                assert(max(abs(squeeze(query.expected_posteriors(phase, observed, :))' - direct)) < 1e-12);
                stats.belief_checks = stats.belief_checks + 1;
            end
            err = abs(query.information_gain(phase) - (H - expected_H));
            stats.max_information_error = max(stats.max_information_error, err);
            assert(err < 1e-12);
            assert(max(abs(expected_b - b)) < 1e-12);
            assert(abs(sum(query.outcome_probs(phase, :)) - 1) < 1e-12);
            stats.query_checks = stats.query_checks + 1;
        end
        [bestG, ~] = min(query.G);
        expected_choice = find(query.G <= bestG + 1e-12, 1, 'first') - 1;
        assert(query.phase == expected_choice);
    end
    % A known code has no epistemic gain. Even at zero cost, STOP wins ties.
    known = [0 0 0 0 0 1];
    for cost = [0, .04, 1]
        cc = c; cc.query_cost = cost;
        q = cpb_select_query(known, cc);
        assert(q.phase == 0 && max(abs(q.information_gain)) < 1e-12);
    end
    % Uninformative reports and a prohibitively costly teacher both stop.
    cc = c; cc.demo_reliability = 1/3;
    q = cpb_select_query(ones(1, 6) / 6, cc);
    assert(q.phase == 0 && max(abs(q.information_gain)) < 1e-12);
    cc = c; cc.query_cost = 2;
    q = cpb_select_query(ones(1, 6) / 6, cc);
    assert(q.phase == 0);
    % Exact utility ties favour STOP; symmetric informative positions pick 1.
    q = cpb_select_query(ones(1, 6) / 6, c);
    assert(q.phase == 1);
    cc = c; cc.query_cost = max(q.information_gain);
    q = cpb_select_query(ones(1, 6) / 6, cc);
    assert(q.phase == 0);
    expect_error(@() cpb_code_update(ones(1, 6) / 6, 1, 4, c, p), 'mpb:ImpossibleObservation');
    expect_error(@() cpb_query_model(0, c), 'cpb:InvalidPhase');
    expect_error(@() cpb_select_query(ones(1, 6), c), 'cpb:InvalidPrior');
    % A perfect teacher leaves exact zero support for inconsistent codes.
    cc = c; cc.demo_reliability = 1;
    u = cpb_code_update(ones(1, 6) / 6, 1, 3, cc, p);
    assert(all(u.posterior(c.routes(:, 1)' ~= 3) == 0));
    assert(abs(sum(u.posterior) - 1) < 1e-12);
    % Code labels can be permuted without changing query utilities.
    permutation = [4 1 6 2 5 3];
    b = [1 2 3 5 8 13] / 32;
    qa = cpb_select_query(b, c);
    cc = c; cc.routes = c.routes(permutation, :);
    qb = cpb_select_query(b(permutation), cc);
    err = max(abs(qa.G - qb.G));
    stats.max_permutation_error = max(stats.max_permutation_error, err);
    assert(err < 1e-12);
    ua = cpb_code_update(b, 2, 3, c, p);
    ub = cpb_code_update(b(permutation), 2, 3, cc, p);
    assert(max(abs(ua.posterior(permutation) - ub.posterior)) < 1e-12);
    % No hard-coded canonical route: clean report histories learn all six,
    % including the two noncanonical/withheld compositions of MPB v1.
    stats.clean_teacher_queries = zeros(1, 6);
    stats.clean_teacher_posterior = zeros(1, 6);
    for truth = 1:6
        b = ones(1, 6) / 6;
        for step = 1:c.max_queries
            query = cpb_select_query(b, c);
            if query.phase == 0, break; end
            observed = c.routes(truth, query.phase); % synthetic teacher only
            u = cpb_code_update(b, query.phase, observed, c, p);
            b = u.posterior;
            stats.clean_teacher_queries(truth) = stats.clean_teacher_queries(truth) + 1;
        end
        [~, predicted] = max(b);
        assert(predicted == truth && b(truth) > .9);
        stats.clean_teacher_posterior(truth) = b(truth);
        stats.learned_programs = stats.learned_programs + 1;
    end
    % Learned cue likelihoods, uniform available-memory prior, code mixture.
    counts = 1 + reshape(mod(1:36, 7), 6, 6);
    counts(1:7:end) = counts(1:7:end) + [20 18 16 14 12 10];
    codes = .03 * ones(6, 6);
    mapping = [6 1 5 2 4 3];
    for slot = 1:6, codes(slot, mapping(slot)) = .85; end
    available = [1 0 1 0 1 0; ones(1, 6)];
    for which = 1:size(available, 1)
        seen = logical(available(which, :));
        for obs = 1:6
            r = cpb_recognize(counts, codes, seen, obs, p);
            assert(all(r.q_context(~seen) == 0));
            assert(abs(sum(r.q_context) - 1) < 1e-12);
            assert(abs(sum(r.q_program) - 1) < 1e-12);
            assert(r.max_error < 1e-12);
            stats.max_recognition_error = max(stats.max_recognition_error, r.max_error);
            slot_perm = [6 2 4 1 5 3];
            rp = cpb_recognize(counts(slot_perm, :), codes(slot_perm, :), seen(slot_perm), obs, p);
            err = max(abs(rp.q_program - r.q_program));
            err = max(err, max(abs(rp.q_context - r.q_context(slot_perm))));
            assert(err < 1e-12);
            cp = [4 1 6 2 5 3];
            rc = cpb_recognize(counts, codes(:, cp), seen, obs, p);
            err = max(err, max(abs(rc.q_program - r.q_program(cp))));
            assert(err < 1e-12);
            stats.max_permutation_error = max(stats.max_permutation_error, err);
            stats.recognition_checks = stats.recognition_checks + 1;
        end
    end
    % Unseen code memory can be initialized to zeros without affecting output.
    seen = logical(available(1, :));
    empty_codes = codes; empty_codes(~seen, :) = 0;
    r = cpb_recognize(counts, codes, seen, 2, p);
    re = cpb_recognize(counts, empty_codes, seen, 2, p);
    assert(isequal(r.q_program, re.q_program));
    expect_error(@() cpb_recognize(counts, codes, false(1, 6), 1, p), 'cpb:InvalidSeen');
    expect_error(@() cpb_recognize(zeros(6), codes, true(1, 6), 1, p), 'cpb:InvalidCueCounts');
    assert(isequal(p, original_p), 'The fixed interpreter parameter struct changed.');
    stats.fixed_weights_unchanged = true;
    stats.passed = true;
    fprintf('CPB core passed: %d Bayes updates, %d query audits, %d retrieval audits.\n', ...
        stats.belief_checks, stats.query_checks, stats.recognition_checks);
    fprintf('  max Bayes %.3g; free energy %.3g; information %.3g; retrieval %.3g.\n', ...
        stats.max_bayes_error, stats.max_free_energy_error, ...
        stats.max_information_error, stats.max_recognition_error);
end

function H = entropy_of(b)
    positive = b > 0;
    H = -sum(b(positive) .* log(b(positive)));
end

function expect_error(f, expected_identifier)
    did_error = false;
    try
        f();
    catch err
        did_error = true;
        assert(strcmp(err.identifier, expected_identifier), ...
            'Expected error %s, got %s.', expected_identifier, err.identifier);
    end
    assert(did_error, 'Expected error %s was not raised.', expected_identifier);
end
