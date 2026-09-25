function out = tva_actor(prior,L,action,observation,payload,p,selector_override)
% Pure actor: no hidden context or target class is accepted as an argument.
% prior: positive 1x3; L: observation x context x action; payload: 3x3 rows.
% selector_override is an explicit fixed-program intervention, empty normally.
if nargin<6 || isempty(p), p=tva_parameters(); end
if nargin<7, selector_override=[]; end
prior=prior(:)';
assert(numel(prior)==3 && all(isfinite(prior)) && all(prior>0), ...
    'tva:Prior','Prior must have three finite positive entries.');
prior=prior/sum(prior);
assert(isequal(size(L),[3 3 3]) && all(isfinite(L(:))) && all(L(:)>0), ...
    'tva:Likelihood','Likelihood must be positive with shape 3x3x3.');
column_errors=abs(sum(L,1)-1);
assert(max(column_errors(:))<1e-12,'tva:Likelihood','Likelihood columns must sum to one.');
assert(isscalar(action) && any(action==1:3) && isscalar(observation) && any(observation==1:3), ...
    'tva:Index','Action and observation indices must be 1, 2 or 3.');
assert(isequal(size(payload),[3 3]) && isreal(payload) && all(isfinite(payload(:))), ...
    'tva:Payload','Payload must be a finite real 3x3 matrix.');
out.prior=prior; out.action=action; out.observation=observation;
out.log_evidence=log(L(observation,:,action));
out.key_inputs=[log(prior(:)) out.log_evidence(:) zeros(3,1)];
out.query_input=ones(1,3);
out.Q=tva_linear(out.query_input,p.Wq,'right');
out.K=tva_linear(out.key_inputs,p.Wk,'right');
out.V=tva_linear(payload,p.Wv,'right');
out.scores=tva_linear(out.Q,out.K','right')/sqrt(3);
out.posterior=tva_softmax(out.scores);
out.program=out.posterior;
if ~isempty(selector_override)
    assert(isscalar(selector_override) && any(selector_override==1:3), ...
        'tva:Selector','Fixed selector must be 1, 2 or 3.');
    out.program=zeros(1,3); out.program(selector_override)=1;
end
out.z=tva_linear(out.V,out.program,'left');
out.bayes_predictive=tva_linear(out.V,out.posterior,'left');
% Three dedicated row-selection branches, followed by posterior mixing.
out.compiled_branches=zeros(3,3);
out.compiled_branches(1,:)=out.V(1,:);
out.compiled_branches(2,:)=out.V(2,:);
out.compiled_branches(3,:)=out.V(3,:);
out.z_compiled=zeros(1,3);
for c=1:3, out.z_compiled=out.z_compiled+out.program(c)*out.compiled_branches(c,:); end
out.readout=tva_readout(out.z,p);
out.pclass=out.readout.pclass;
end
