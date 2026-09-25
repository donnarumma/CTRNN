function [posterior, info] = bci_spm_reference(likelihood_high, observed_index, prior)
%BCI_SPM_REFERENCE SPM12 reference for a single binary Bayesian update.
% [q, info] = bci_spm_reference([P(high|H1), P(high|H2)], observation, prior)
% observation is 1 for high and 2 for low. Default prior is [.5; .5].
% The original spm_MDP_VB_X and its dependencies must already be on path.
%
% The solver runs in hidden-Markov-model mode with one observation, without
% policies, actions, rewards, generated hidden states, or a learned posterior
% supplied as input. Its only inputs are the likelihood, prior and observed
% one-hot sensory outcome. With T=1, X(:,1) is the online posterior; there is
% no future observation to smooth it. SPM's HMM return omits the field xn.
%
% tau=1 makes this one-factor static log-belief update converge directly.
% It changes the numerical relaxation rate, not the Bayesian model. SPM's
% default tau=4 and fixed 16 iterations otherwise leave a small residual.
% This function is a numerical reference ONLY: it does not drive the CTRNN.

if nargin < 3 || isempty(prior), prior = [.5; .5]; end
assert(isnumeric(likelihood_high) && isreal(likelihood_high) && ...
    numel(likelihood_high) == 2 && all(isfinite(likelihood_high(:))) && ...
    all(likelihood_high(:) > 0 & likelihood_high(:) < 1), ...
    'bci:Likelihood', 'likelihood_high must contain two probabilities in (0,1).');
assert(isnumeric(observed_index) && isscalar(observed_index) && ...
    any(observed_index == [1 2]), ...
    'bci:Observation', 'observed_index must be 1 (high) or 2 (low).');
assert(isnumeric(prior) && isreal(prior) && numel(prior) == 2 && ...
    all(isfinite(prior(:))) && all(prior(:) > 0), ...
    'bci:Prior', 'prior must contain two finite positive entries.');
assert(exist('spm_MDP_VB_X', 'file') ~= 0, 'bci:SPMNotFound', ...
    'Original SPM12 spm_MDP_VB_X must be available on the MATLAB/Octave path.');

likelihood_high = double(likelihood_high(:)');
prior = double(prior(:));
prior = prior / sum(prior);
likelihood = [likelihood_high; 1-likelihood_high];
outcome = zeros(2,1);
outcome(observed_index) = 1;

mdp.A = {likelihood};
mdp.B = {eye(2)}; % Required HMM schema; no transition occurs at T=1.
mdp.D = {prior};
mdp.O = {outcome};
mdp.o = 0;       % T and HMM mode are derived from the supplied O.
mdp.tau = 1;
mdp.eta = 0;
solved = spm_MDP_VB_X(mdp, struct('plot', 0));
posterior = full(solved.X{1}(:,1));

analytic = likelihood(observed_index,:)' .* prior;
analytic = analytic / sum(analytic);
absolute_error = max(abs(posterior - analytic));
assert(solved.T == 1 && isempty(solved.u), 'bci:SPMMode', ...
    'SPM reference must have one observation and no actions.');
assert(absolute_error < 1e-7, 'bci:SPMReferenceMismatch', ...
    'SPM and analytic Bayes differ by %.3g.', absolute_error);

info.solver = which('spm_MDP_VB_X');
info.mode = 'HMM, one observation, no policies or actions';
info.observed_index = observed_index;
info.likelihood = likelihood;
info.prior = prior;
info.posterior_analytic = analytic;
info.max_absolute_error = absolute_error;
info.update_time_constant = mdp.tau;
info.number_of_observations = solved.T;
info.number_of_actions = numel(solved.u);
info.posterior_source = 'X{1}(:,1): online because T=1; no future evidence';
end
