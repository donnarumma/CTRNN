function report = ai2_test_backend(cfg)
%AI2_TEST_BACKEND Numerical checks of code-as-input emulation and probe design.
if nargin < 1, cfg = struct(); end
bank = ai2_build_bank(cfg);
cfg = bank.cfg;
assert(bank.interpreter.numUnits == 14 && bank.interpreter.numExternalInput == 6);
assert(bank.meta.fixed_internal_matrix && bank.meta.fixed_external_matrix);
assert(all(abs(diff(bank.programs, 1, 1)) > 0));
assert(bank.meta.target_sensor_separation(1) < 1e-10);
assert(bank.meta.target_sensor_separation(2) > 0.4);
assert(all(bank.meta.rmse(:) < 0.04), 'ai2:Approximation', 'Probe rollout RMSE exceeded 0.04.');
assert(all(bank.meta.tail_range(:) < 0.001), 'ai2:Settling', 'Interpreter did not settle.');
report.probe_rmse = bank.meta.rmse;
report.probe_max_error = bank.meta.max_error;
report.probe_steady_error = bank.meta.steady_error;
report.target_sensor_separation = bank.meta.target_sensor_separation;
report.interpreter_sensor_separation = bank.meta.interpreter_sensor_separation;
report.interpreter_neutral_model_bias = bank.interpreter_predictions(:, 1, 1) - bank.target_predictions(:, 1, 1);
report.fixed_internal_matrix = bank.meta.fixed_internal_matrix;
report.fixed_external_matrix = bank.meta.fixed_external_matrix;
report.target_predictions = bank.target_predictions;
report.interpreter_predictions = bank.interpreter_predictions;
report.code = bank.programs;
% This physical input sequence is not used to construct the supervisor's A.
levels = [0.5 0.5; 0.9 0.5; 0.2 0.8; 0.7 0.3];
block = max(1, round(15 / cfg.dt));
physicalInput = kron(levels, ones(block, 1));
for context = 1:2
    target = ai2_simulate(bank.target_nets{context}, physicalInput, [], cfg);
    interpreted = ai2_simulate(bank.interpreter, physicalInput, bank.programs(context, :), cfg);
    wrong = ai2_simulate(bank.interpreter, physicalInput, bank.programs(3-context, :), cfg);
    difference = target.readout - interpreted.readout;
    wrongDifference = target.readout - wrong.readout;
    report.heldout_rmse(context) = sqrt(mean(difference(:).^2));
    report.wrong_code_rmse(context) = sqrt(mean(wrongDifference(:).^2));
    assert(report.heldout_rmse(context) < 0.05, 'ai2:Heldout', 'Held-out RMSE exceeded 0.05.');
    assert(report.wrong_code_rmse(context) > 3 * report.heldout_rmse(context), ...
        'ai2:CodeEffect', 'The correct program did not materially improve the rollout.');
end
% Check discretization separately from the multiplier approximation.
fineCfg = cfg;
fineCfg.dt = cfg.dt / 2;
fine = ai2_build_bank(fineCfg);
report.half_dt_steady_change = max(abs(bank.interpreter_predictions(:) - fine.interpreter_predictions(:)));
assert(report.half_dt_steady_change < 0.001, 'ai2:Discretization', 'Halving dt changed steady predictions materially.');
% Time-scale perturbation is a sensitivity check, not proof for all parameters.
slowCfg = cfg;
slowCfg.multiplier_tau = cfg.multiplier_tau * 2;
slow = ai2_build_bank(slowCfg);
report.slower_multiplier_max_rmse = max(slow.meta.rmse(:));
report.slower_multiplier_steady_change = max(abs(bank.interpreter_predictions(:) - slow.interpreter_predictions(:)));
report.scope = ['Tests cover two contracting targets, two constant probes and one held-out piecewise input. ' ...
    'No limit-cycle accuracy or continuous online program switching claim is made.'];
report.passed = true;
fprintf('CTRNN backend: PASS. Probe max RMSE %.6f; held-out RMSE %.6f / %.6f.\n', ...
    max(report.probe_rmse(:)), report.heldout_rmse);
fprintf('Wrong-code RMSE %.6f / %.6f; half-dt steady change %.3g.\n', ...
    report.wrong_code_rmse, report.half_dt_steady_change);
end
