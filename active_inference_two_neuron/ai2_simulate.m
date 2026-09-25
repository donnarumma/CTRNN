function sim = ai2_simulate(net, input, program, cfg)
%AI2_SIMULATE Deterministic rollout through the repository's runCTRNN.
% sim = ai2_simulate(net, input, program, cfg)
% input is either one 1x2 constant probe or an Nx2 physical-input sequence.
% program is [] for a target, or a 1x4 constant interpreter program.
% The state convention follows runCTRNN_NoAsymWay: output-space dynamics.
if nargin < 3, program = []; end
if nargin < 4, cfg = struct(); end
if ~isfield(cfg, 'duration'), cfg.duration = 60; end
if ~isfield(cfg, 'dt'), cfg.dt = net.dt; end
validateattributes(input, {'numeric'}, {'2d','real','finite'});
assert(size(input, 2) == 2, 'ai2:PhysicalInputDimension', 'Physical input must have two columns.');
validateattributes(cfg.dt, {'numeric'}, {'scalar','positive','finite'});
validateattributes(cfg.duration, {'numeric'}, {'scalar','positive','finite'});
assert(all(input(:) >= 0 & input(:) <= 1), ...
    'ai2:InputRange', 'Multiplier inputs must remain in [0,1].');
if size(input, 1) == 1
    input = repmat(input, max(1, round(cfg.duration / cfg.dt)), 1);
end
if ~isempty(program)
    validateattributes(program, {'numeric'}, {'row','numel',4,'real','finite'});
    assert(all(program >= 0 & program <= 1), ...
        'ai2:ProgramRange', 'Encoded weights must remain in [0,1].');
    externalInput = [input, repmat(program, size(input, 1), 1)];
else
    externalInput = input;
end
assert(size(externalInput, 2) == net.numExternalInput, ...
    'ai2:InputDimension', 'Physical input plus code has the wrong dimension.');
assert(cfg.dt <= min(net.tau), 'ai2:TimeStep', ...
    'Euler dt must not exceed the smallest time constant.');
net.dt = cfg.dt;
net.externalInput = externalInput;
net.asympMod = false;
% NET_Noise otherwise calls normrnd even at zero variance in this repository.
net.NOISE = @(n) zeros(1, n.numUnits);
sim.output = runCTRNN(net);
sim.readout = sim.output(:, 1:2);
sim.time = (1:size(input, 1))' * cfg.dt;
sim.input = input;
sim.program = program;
tail = max(1, floor(0.9 * size(input, 1))):size(input, 1);
sim.steady = mean(sim.readout(tail, :), 1);
sim.tail_range = max(sim.readout(tail, :), [], 1) - min(sim.readout(tail, :), [], 1);
end
