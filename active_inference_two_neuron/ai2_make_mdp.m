function [mdp, design] = ai2_make_mdp(bank, truth, condition, cfg)
% Discrete SPM supervisor over a continuous CTRNN interpreter.
% Factor 1: hidden target program. Factor 2: experimental mode.
% A is the real process (compiled target); a encodes the interpreter model.
assert(ismember(truth, [1 2]), 'ai2:Truth', 'truth must be 1 or 2.');
assert(any(strcmp(condition, {'active', 'passive', 'oracle'})), ...
    'ai2:Condition', 'Unknown experimental condition.');
assert(size(bank.probes,1) == 2, 'ai2:Probes', 'Expected neutral and informative probes.');
design.mode_names = {'start', 'neutral', 'informative', 'execute_1', 'execute_2'};
design.action_names = {'neutral', 'informative', 'execute_1', 'execute_2'};
design.sensor_names = {'none', 'low', 'high'};
design.feedback_names = {'none', 'match', 'mismatch'};
design.process_high = normal_high(bank.target_predictions(:,:,1), cfg);
design.model_high = normal_high(bank.interpreter_predictions(:,:,1), cfg);
process = make_likelihood(design.process_high);
model = make_likelihood(design.model_high);
mdp.A = process;
for g = 1:numel(model)
    mdp.a{g} = cfg.model_concentration * model{g};
end
mdp.B{1} = eye(2);
mdp.B{2} = zeros(5,5,4);
for action = 1:4
    mdp.B{2}(action + 1,:,action) = 1;
end
mdp.D{1} = [.5; .5];
if strcmp(condition, 'oracle')
    mdp.D{1} = double((1:2)' == truth);
end
mdp.D{2} = [1;0;0;0;0];
controls = [1 1 2 2; 3 4 3 4];
if strcmp(condition, 'passive'), controls = controls(:,1:2); end
mdp.V = ones(2, size(controls,2), 2);
mdp.V(:,:,2) = controls;
mdp.T = 3;
mdp.C{1} = zeros(3,3);
mdp.C{2} = zeros(3,3);
mdp.C{2}(:,3) = [0;cfg.reward;-cfg.reward];
mdp.C{3} = zeros(5,3);
mdp.C{3}(3,2) = -cfg.probe_cost;
mdp.s = [truth;1]; % This is available ONLY to the generative process.
mdp.alpha = cfg.action_precision;
mdp.beta = 1;
mdp.eta = 0;       % No learning of the already calibrated model.
mdp.label.factor = {'program','mode'};
mdp.label.name = {{'program_1','program_2'}, design.mode_names};
mdp.label.modality = {'sensor','feedback','mode'};
mdp.label.outcome = {design.sensor_names,design.feedback_names,design.mode_names};
design.model = model;
design.process = process;
design.controls = controls;
end

function p = normal_high(mu, cfg)
% P(y + epsilon > threshold), epsilon ~ N(0,sigma^2); no statistics toolbox.
p = .5 * erfc((cfg.sensor_threshold - mu) ./ (sqrt(2)*cfg.sensor_sigma));
p = min(max(p,1e-12),1-1e-12);
end

function A = make_likelihood(high)
A{1} = zeros(3,2,5); % Observation of first neural activity, binned low/high.
A{2} = zeros(3,2,5); % Final match/mismatch of selected and hidden program.
A{3} = zeros(5,2,5); % Observable experimental mode (proprioceptive channel).
for context = 1:2
    for mode = 1:5
        A{3}(mode,context,mode) = 1;
        if mode == 2 || mode == 3
            p = high(context,mode-1);
            A{1}(:,context,mode) = [0;1-p;p];
        else
            A{1}(1,context,mode) = 1;
        end
        if mode < 4
            A{2}(1,context,mode) = 1;
        else
            correct = (mode-3 == context);
            A{2}(3-correct,context,mode) = 1;
        end
    end
end
end
