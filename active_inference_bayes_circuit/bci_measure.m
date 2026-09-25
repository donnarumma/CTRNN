function out = bci_measure(target,interpreted,likelihood,prior)
% Evaluate outputs; Bayes and normalization here NEVER enter the neural input.
% likelihood = [P(observed|H1),P(observed|H2)].
joint = likelihood.*prior;
posterior = joint/sum(joint);
out.bayes = posterior;
out.raw_final = interpreted.readout(end,:);
out.raw_steady = interpreted.steady;
out.target_final = target.readout(end,:);
out.decoded_final = out.raw_final/sum(out.raw_final);
out.raw_error = max(abs(out.raw_final-posterior));
out.target_error = max(abs(out.target_final-posterior));
out.decoded_error = max(abs(out.decoded_final-posterior));
out.normalization_error = max(abs(sum(interpreted.readout,2)-1));
out.final_normalization_error = abs(sum(out.raw_final)-1);
out.tail_range = max(interpreted.tail_range);
delta = interpreted.readout-target.readout;
out.emulation_rmse = sqrt(mean(delta(:).^2));
% All probability diagnostics normalize the readout externally, for analysis.
% Raw outputs and their normalization defect remain explicit above.
raw = [prior;interpreted.readout];
q = bsxfun(@rdivide,raw,sum(raw,2));
q_target = [prior;target.readout];
q_target = bsxfun(@rdivide,q_target,sum(q_target,2));
out.kl_curve = sum(q.*log(bsxfun(@rdivide,q,posterior)),2);
out.target_kl_curve = sum(q_target.*log(bsxfun(@rdivide,q_target,posterior)),2);
out.free_energy = out.kl_curve-log(sum(joint));
out.target_free_energy = out.target_kl_curve-log(sum(joint));
out.final_kl = out.kl_curve(end);
increments = diff(out.free_energy);
out.max_free_energy_increase = max([0;increments]);
out.total_free_energy_increase = sum(max(0,increments));
out.target_max_free_energy_increase = max([0;diff(out.target_free_energy)]);
out.decoded_trajectory = q;
end
