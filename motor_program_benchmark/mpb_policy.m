function out = mpb_policy(b, Lprobe, terminal_category, execution_cost, look_cost, failure_penalty, mode)
%MPB_POLICY Exact finite-horizon policy selection for the motor benchmark.
%   B is a 1-by-6 belief over context (the required route), not a true
%   context. LPROBE(o,r) is the probability of probe outcome o in context r.
%   TERMINAL_CATEGORY(route,r) is 1 (success), 2 (first touch wrong), or 3
%   (second touch wrong). EXECUTION_COST(route,r) includes execution only.
%   The policies are six complete route commitments and one LOOK followed
%   by an optimally contingent route. The common outcome space consists
%   of 7 cue codes (0 means no LOOK), each with 3 terminal categories.
%
%   Active G = expected failure penalty + expected execution/LOOK cost
%              - I(context; complete outcome) + log Z_preferences.
%   In 'no_epistemic' mode the information-gain coefficient is zero, but
%   LOOK remains available and can be selected for its instrumental value.
%   For active mode the equivalent risk-plus-ambiguity expression is
%   evaluated independently for every complete candidate policy.

if nargin < 7, mode = 'active'; end
validate_inputs(b, Lprobe, terminal_category, execution_cost, look_cost, failure_penalty, mode);
b = reshape(b, 1, 6);
epistemic_weight = double(strcmp(mode, 'active'));
log_Z = log(7) + log(1 + 2 * exp(-failure_penalty));
log_C = repmat([-log_Z, -failure_penalty-log_Z, -failure_penalty-log_Z], 1, 7);

G = zeros(1, 7);
information_gain = zeros(1, 7);
expected_failure = zeros(1, 7);
expected_execution_cost = zeros(1, 7);
risk_ambiguity_G = zeros(1, 7);
joint_distributions = cell(1, 7);
for route = 1:6
    joint = commit_joint(b, terminal_category(route, :));
    exec_cost = sum(b .* execution_cost(route, :));
    s = score_candidate(joint, b, exec_cost, 0, failure_penalty, log_C, epistemic_weight);
    G(route) = s.G;
    information_gain(route) = s.information_gain;
    expected_failure(route) = s.expected_failure;
    expected_execution_cost(route) = exec_cost;
    risk_ambiguity_G(route) = s.risk_ambiguity_G;
    joint_distributions{route} = joint;
end

% Mutual-information chain rule separates the contingent optimization:
% I(r; o_probe, o_terminal) = I(r; o_probe)
%                            + E_o I(r; o_terminal | o_probe=o).
% The first term does not depend on the child routes. Each reachable
% observation can therefore choose its minimum-G commitment independently.
observation_probability = (Lprobe * b')';
contingent_routes = ones(1, 6);
conditional_commit_G = nan(6, 6);
for observation = 1:6
    if observation_probability(observation) > 0
        posterior = b .* Lprobe(observation, :) / observation_probability(observation);
        for route = 1:6
            joint = commit_joint(posterior, terminal_category(route, :));
            exec_cost = sum(posterior .* execution_cost(route, :));
            s = score_candidate(joint, posterior, exec_cost, 0, failure_penalty, log_C, epistemic_weight);
            conditional_commit_G(observation, route) = s.G;
        end
        contingent_routes(observation) = first_minimum(conditional_commit_G(observation, :));
    end
end

joint = zeros(6, 21);
exec_cost = 0;
for observation = 1:6
    route = contingent_routes(observation);
    for context = 1:6
        probability = b(context) * Lprobe(observation, context);
        outcome = 3 * observation + terminal_category(route, context);
        joint(context, outcome) = joint(context, outcome) + probability;
        exec_cost = exec_cost + probability * execution_cost(route, context);
    end
end
s = score_candidate(joint, b, exec_cost, look_cost, failure_penalty, log_C, epistemic_weight);
G(7) = s.G;
information_gain(7) = s.information_gain;
expected_failure(7) = s.expected_failure;
expected_execution_cost(7) = exec_cost;
risk_ambiguity_G(7) = s.risk_ambiguity_G;
joint_distributions{7} = joint;

choice = first_minimum(G);
out.choice = choice;
out.use_look = choice == 7;
out.chosen_route = choice * double(choice ~= 7);
out.contingent_routes = contingent_routes;
out.commit_G = G(1:6);
out.look_G = G(7);
out.policy_G = G;
out.information_gain = information_gain;
out.expected_failure = expected_failure;
out.expected_execution_cost = expected_execution_cost;
out.expected_cost = expected_execution_cost + [zeros(1, 6), look_cost];
out.choice_expected_success = 1 - expected_failure(choice);
out.choice_expected_cost = out.expected_cost(choice);
out.normalization_constant = log_Z;
out.log_preference_normalizer = log_Z;
out.risk_ambiguity_G = risk_ambiguity_G;
out.joint_distributions = joint_distributions;
out.observation_probability = observation_probability;
out.conditional_commit_G = conditional_commit_G;
out.epistemic_weight = epistemic_weight;
out.mode = mode;
end

function joint = commit_joint(b, categories)
joint = zeros(6, 21);
for context = 1:6
    joint(context, categories(context)) = b(context);
end
end

function s = score_candidate(joint, b, execution_cost, look_cost, penalty, log_C, epistemic_weight)
q_outcome = sum(joint, 1);
information_gain = 0;
ambiguity = 0;
for context = 1:6
    positive = joint(context, :) > 0;
    if b(context) > 0 && any(positive)
        joint_positive = joint(context, positive);
        log_conditional = log(joint_positive / b(context));
        information_gain = information_gain + sum(joint_positive .* (log_conditional - log(q_outcome(positive))));
        ambiguity = ambiguity - sum(joint_positive .* log_conditional);
    end
end
positive = q_outcome > 0;
risk = sum(q_outcome(positive) .* (log(q_outcome(positive)) - log_C(positive)));
failure_mask = repmat([false, true, true], 1, 7);
s.expected_failure = sum(q_outcome(failure_mask));
s.information_gain = information_gain;
s.risk_ambiguity_G = risk + ambiguity + execution_cost + look_cost;
% log C(success) = -log Z. This evaluation is independent of the entropy
% calculation above, permitting an exact risk/ambiguity identity check.
s.G = penalty * s.expected_failure + execution_cost + look_cost ...
    - epistemic_weight * information_gain - log_C(1);
end

function index = first_minimum(values)
% Stable preference for the lowest-index action within an absolute 1e-12.
index = find(values <= min(values) + 1e-12, 1, 'first');
end

function validate_inputs(b, Lprobe, categories, costs, look_cost, penalty, mode)
assert(isnumeric(b) && numel(b) == 6 && all(isfinite(b(:))) && all(b(:) >= 0), ...
    'mpb_policy:belief', 'Belief must contain six finite nonnegative entries.');
assert(abs(sum(b(:)) - 1) < 1e-10, 'mpb_policy:beliefMass', 'Belief must sum to one.');
assert(isequal(size(Lprobe), [6, 6]) && all(isfinite(Lprobe(:))) && all(Lprobe(:) >= 0), ...
    'mpb_policy:probe', 'Probe likelihood must be a finite nonnegative 6-by-6 matrix.');
assert(all(abs(sum(Lprobe, 1)-1) < 1e-10), 'mpb_policy:probeMass', 'Every probe-likelihood column must sum to one.');
assert(isequal(size(categories), [6, 6]) && all(ismember(categories(:), 1:3)), ...
    'mpb_policy:categories', 'Terminal categories must be a 6-by-6 matrix with values 1, 2, or 3.');
assert(isequal(size(costs), [6, 6]) && all(isfinite(costs(:))) && all(costs(:) >= 0), ...
    'mpb_policy:costs', 'Execution costs must be finite and nonnegative.');
assert(isscalar(look_cost) && isfinite(look_cost) && look_cost >= 0, ...
    'mpb_policy:lookCost', 'LOOK cost must be finite and nonnegative.');
assert(isscalar(penalty) && isfinite(penalty) && penalty >= 0, ...
    'mpb_policy:penalty', 'Failure penalty must be finite and nonnegative.');
assert(ischar(mode) && any(strcmp(mode, {'active', 'no_epistemic'})), ...
    'mpb_policy:mode', 'Mode must be active or no_epistemic.');
end
