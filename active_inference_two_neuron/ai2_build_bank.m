function bank = ai2_build_bank(cfg)
%AI2_BUILD_BANK Two target CTRNNs sharing one 14-neuron interpreter.
% This is a finite, restricted family, not a claim of universal emulation.
% Programs encode [W11 W22 E11 E22] with weights2program in [-5,5].
% W12=W21=E12=E21=0; biases, time constants, and readout are fixed.
% bank.target_predictions and bank.interpreter_predictions are [context,probe,neuron].
if nargin < 1, cfg = struct(); end
cfg = defaults(cfg, 'dt', 0.05, 'duration', 60, ...
    'target_tau', 5, 'multiplier_tau', 0.25, 'weight_range', [-5 5]);
if ~isfield(cfg, 'ctrnn_root') || isempty(cfg.ctrnn_root)
    cfg.ctrnn_root = getenv('CTRNN_ROOT');
    if isempty(cfg.ctrnn_root)
        candidate = fileparts(fileparts(mfilename('fullpath')));
        if exist(fullfile(candidate, 'mulation.m'), 'file')
            cfg.ctrnn_root = candidate;
        else
            cfg.ctrnn_root = fullfile(getenv('HOME'), 'tools', 'CTRNN');
        end
    end
end
assert(exist(fullfile(cfg.ctrnn_root, 'mulation.m'), 'file') == 2, ...
    'ai2:MissingCTRNN', 'Set cfg.ctrnn_root or CTRNN_ROOT to the CTRNN repository.');
% Do not use genpath: historical examples contain conflicting sigmoid.m files.
addpath(cfg.ctrnn_root);
validateattributes(cfg.target_tau, {'numeric'}, {'scalar','positive','finite'});
validateattributes(cfg.multiplier_tau, {'numeric'}, {'scalar','positive','finite'});
validateattributes(cfg.weight_range, {'numeric'}, {'row','numel',2,'real','finite'});
assert(cfg.weight_range(1) < cfg.weight_range(2), 'ai2:WeightRange', 'Invalid weight range.');
assert(cfg.weight_range(1) <= -2 && cfg.weight_range(2) >= 2, ...
    'ai2:WeightRange', 'The range must contain both -2 and +2.');
weights = [2 -2 -2 2; -2 2 2 -2];
for context = 1:2
    net = createCTRNN(2, 2, [0 0], diag(weights(context, 1:2)), diag(weights(context, 3:4)));
    net.tau = cfg.target_tau * ones(1, 2);
    net.dt = cfg.dt;
    net.initialOutValues = [0.5 0.5];
    net.externalInput = [0.5 0.5];
    net.NOISE = @(n) zeros(1, n.numUnits);
    bank.target_nets{context} = net;
end
bank.programs = weights2program(weights, cfg.weight_range(1), cfg.weight_range(2));
bank.interpreter = compile(bank.target_nets{1}, cfg);
comparison = compile(bank.target_nets{2}, cfg);
bank.meta.fixed_internal_matrix = isequal(bank.interpreter.internalWMatrix, comparison.internalWMatrix);
bank.meta.fixed_external_matrix = isequal(bank.interpreter.externalWMatrix, comparison.externalWMatrix);
assert(bank.meta.fixed_internal_matrix && bank.meta.fixed_external_matrix, ...
    'ai2:ChangingHardware', 'Target compilation changed the interpreter matrices.');
bank.cfg = cfg;
bank.probes = [0.5 0.5; 0.9 0.5];
bank.probe_names = {'neutral', 'neuron_1_probe'};
bank.context_names = {'self_exciting_neuron_1', 'self_inhibiting_neuron_1'};
bank.program_names = {'program_1', 'program_2'};
bank.target_predictions = zeros(2, 2, 2);
bank.interpreter_predictions = zeros(2, 2, 2);
bank.meta.rmse = zeros(2, 2);
bank.meta.max_error = zeros(2, 2);
bank.meta.steady_error = zeros(2, 2);
bank.meta.tail_range = zeros(2, 2);
for context = 1:2
    for probe = 1:2
        target = ai2_simulate(bank.target_nets{context}, bank.probes(probe, :), [], cfg);
        interpreted = ai2_simulate(bank.interpreter, bank.probes(probe, :), bank.programs(context, :), cfg);
        bank.target_predictions(context, probe, :) = target.steady;
        bank.interpreter_predictions(context, probe, :) = interpreted.steady;
        bank.target_rollouts{context, probe} = target;
        bank.interpreter_rollouts{context, probe} = interpreted;
        difference = target.readout - interpreted.readout;
        bank.meta.rmse(context, probe) = sqrt(mean(difference(:).^2));
        bank.meta.max_error(context, probe) = max(abs(difference(:)));
        bank.meta.steady_error(context, probe) = max(abs(target.steady - interpreted.steady));
        bank.meta.tail_range(context, probe) = max(interpreted.tail_range);
    end
end
bank.meta.cfg = cfg;
bank.meta.program_columns = {'W11','W22','E11','E22'};
bank.meta.program_range = cfg.weight_range;
bank.meta.readout_neurons = [1 2];
bank.meta.sensor_neuron = 1;
bank.meta.n_target_neurons = 2;
bank.meta.n_interpreter_neurons = bank.interpreter.numUnits;
bank.meta.target_sensor_separation = abs(diff(bank.target_predictions(:, :, 1), 1, 1));
bank.meta.interpreter_sensor_separation = abs(diff(bank.interpreter_predictions(:, :, 1), 1, 1));
bank.meta.limitations = ['Finite two-program family with fixed topology, biases and time constants. ' ...
    'Historical multiplication motifs are approximate: neutral probes can acquire spurious context evidence. ' ...
    'Targets are contracting fixed-point systems; limit-cycle emulation is not established.'];
end

function interpreted = compile(net, cfg)
interpreted = net;
interpreted = mulation(interpreted, 'internal', 1, 1, cfg.weight_range, cfg.multiplier_tau);
interpreted = mulation(interpreted, 'internal', 2, 2, cfg.weight_range, cfg.multiplier_tau);
interpreted = mulation(interpreted, 'external', 1, 1, cfg.weight_range, cfg.multiplier_tau);
interpreted = mulation(interpreted, 'external', 2, 2, cfg.weight_range, cfg.multiplier_tau);
end

function cfg = defaults(cfg, varargin)
for k = 1:2:numel(varargin)
    if ~isfield(cfg, varargin{k}), cfg.(varargin{k}) = varargin{k + 1}; end
end
end
