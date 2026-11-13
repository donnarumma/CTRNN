function out = cpb_code_update(b, phase, observed, c, p)
%CPB_CODE_UPDATE Learn a code posterior from one noisy teacher report.
% The fixed shared-X attention head performs the categorical Bayes update.
% The arguments deliberately exclude task identity and hidden route code.
    if nargin < 5, p = mpb_parameters(); end
    L = cpb_query_model(phase, c);
    out = mpb_belief_update(b, L, observed, p);
    out.phase = phase;
    out.likelihood = L;
end
