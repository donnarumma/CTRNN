function report = test_mpb_policy()
%TEST_MPB_POLICY Independent checks of the finite contingent EFE evaluator.
routes = sortrows(perms(1:3));
category = ones(6, 6);
for action = 1:6
    for context = 1:6
        if routes(action, 1) ~= routes(context, 1)
            category(action, context) = 2;
        elseif routes(action, 2) ~= routes(context, 2)
            category(action, context) = 3;
        end
    end
end
b = ones(1, 6) / 6;
cost = 0.01 * (1 + double(category ~= 2) + double(category == 1));
perfect = eye(6);
active = mpb_policy(b, perfect, category, cost, 0.1, 3, 'active');
instrumental = mpb_policy(b, perfect, category, cost, 0.1, 3, 'no_epistemic');
assert(active.use_look && instrumental.use_look, 'The no-epistemic agent must still be able to purchase useful information.');
assert(isequal(active.contingent_routes, 1:6), 'Perfect observations must select the matching route.');
assert(abs(active.choice_expected_success - 1) < 1e-12);
assert(abs(active.choice_expected_cost - 0.13) < 1e-12);
assert(abs(active.information_gain(7) - log(6)) < 1e-12);

% Execution feedback is itself informative and must enter commitment EFE.
commit_entropy = -sum([1/6, 4/6, 1/6] .* log([1/6, 4/6, 1/6]));
assert(max(abs(active.information_gain(1:6) - commit_entropy)) < 1e-12);
assert(all(active.information_gain >= -1e-12));
identity_error = max(abs(active.policy_G - active.risk_ambiguity_G));
assert(identity_error < 1e-12, 'Active EFE must equal risk plus ambiguity plus action costs.');
assert(max(abs(instrumental.policy_G - instrumental.risk_ambiguity_G - instrumental.information_gain)) < 1e-12);

% Positive-cost, completely uninformative LOOK cannot improve the best
% commitment (including the information produced by terminal feedback).
uninformative = mpb_policy(b, ones(6)/6, category, cost, 0.2, 3, 'active');
assert(~uninformative.use_look);
assert(abs(uninformative.look_G - min(uninformative.commit_G) - 0.2) < 1e-12);
assert(abs(uninformative.information_gain(7) - uninformative.information_gain(1)) < 1e-12);
high_cost = mpb_policy(b, perfect, category, cost, 100, 3, 'active');
assert(~high_cost.use_look);

% With flat terminal preferences and free execution, only information gain
% can justify buying this probe: a direct check of the epistemic ablation.
epistemic = mpb_policy(b, perfect, category, zeros(6), 0.1, 0, 'active');
no_epistemic = mpb_policy(b, perfect, category, zeros(6), 0.1, 0, 'no_epistemic');
assert(epistemic.use_look && ~no_epistemic.use_look);
assert(no_epistemic.choice == 1, 'Ties must prefer the first route.');

% Zero probabilities and unreachable observations must produce finite G.
zero_support = mpb_policy([0.7, 0.3, 0, 0, 0, 0], perfect, category, cost, 0.1, 3, 'active');
assert(all(isfinite(zero_support.policy_G)));
assert(max(abs(zero_support.policy_G-zero_support.risk_ambiguity_G)) < 1e-12);
for candidate = 1:7
    joint = zero_support.joint_distributions{candidate};
    assert(abs(sum(joint(:))-1) < 1e-12);
    assert(max(abs(sum(joint, 2)'-[0.7, 0.3, 0, 0, 0, 0])) < 1e-12);
end

% Exhaust every contingent mapping for an independent two-observation
% likelihood: 6^2=36 distinct trees. Other observation rows are unreachable.
two_obs = zeros(6);
two_obs(1, :) = [0.91, 0.76, 0.6, 0.41, 0.22, 0.08];
two_obs(2, :) = 1-two_obs(1, :);
nonuniform = [0.08, 0.11, 0.17, 0.24, 0.18, 0.22];
asymmetric_cost = reshape(1:36, 6, 6) / 113;
brute_force_error = 0;
for mode_cell = {'active', 'no_epistemic'}
    mode = mode_cell{1};
    out = mpb_policy(nonuniform, two_obs, category, asymmetric_cost, 0.17, 1.3, mode);
    full_tree_values = zeros(6, 6);
    for route1 = 1:6
        for route2 = 1:6
            full_tree_values(route1, route2) = independently_score_tree(nonuniform, two_obs, ...
                category, asymmetric_cost, [route1, route2, 1, 1, 1, 1], 0.17, 1.3, strcmp(mode, 'active'));
        end
    end
    this_error = abs(out.look_G-min(full_tree_values(:)));
    brute_force_error = max(brute_force_error, this_error);
    assert(this_error < 1e-12, 'Conditional EFE optimization must equal exhaustive tree search.');
    direct_value = full_tree_values(out.contingent_routes(1), out.contingent_routes(2));
    assert(abs(out.look_G-direct_value) < 1e-12);
end

% A known context needs no additional observation and receives no IG from
% either a probe or deterministic terminal feedback.
certain = mpb_policy([0, 0, 1, 0, 0, 0], perfect, category, cost, 0.1, 3, 'active');
assert(certain.choice == 3 && ~certain.use_look);
assert(max(abs(certain.information_gain)) < 1e-12);

report.passed = true;
report.risk_ambiguity_error = identity_error;
report.brute_force_error = brute_force_error;
report.exhaustive_trees_per_mode = 36;
report.commit_feedback_information = commit_entropy;
report.look_information = active.information_gain(7);
fprintf('mpb_policy tests passed: risk/ambiguity %.3g, exhaustive trees %.3g.\n', identity_error, brute_force_error);
end

function G = independently_score_tree(b, L, categories, costs, mapping, look_cost, penalty, use_information)
% An independent implementation uses H(r)-H(r|O), rather than the policy
% evaluator's sum over log conditional-to-marginal outcome probabilities.
j = zeros(6, 21);
expected_cost = look_cost;
expected_failure = 0;
for o = 1:6
    for r = 1:6
        probability = b(r)*L(o, r);
        cat = categories(mapping(o), r);
        j(r, 3*o+cat) = probability;
        expected_cost = expected_cost + probability*costs(mapping(o), r);
        expected_failure = expected_failure + probability*(cat ~= 1);
    end
end
positive = b > 0;
context_entropy = -sum(b(positive).*log(b(positive)));
conditional_entropy = 0;
for outcome = 1:21
    mass = sum(j(:, outcome));
    if mass > 0
        posterior = j(:, outcome)/mass;
        positive = posterior > 0;
        conditional_entropy = conditional_entropy - mass*sum(posterior(positive).*log(posterior(positive)));
    end
end
G = penalty*expected_failure + expected_cost ...
    - use_information*(context_entropy-conditional_entropy) + log(7*(1+2*exp(-penalty)));
end
