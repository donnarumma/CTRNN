function policy = tva_policy(prior,L,costs,condition)
% One-step epistemic active inference. No true context or payload is supplied.
% C(o,s)=exp(-cost_s)/(3*sum_b exp(-cost_b)); s reports the chosen action.
if nargin<4, condition='active'; end
prior=prior(:)'; prior=prior/sum(prior); costs=costs(:)';
assert(numel(prior)==3 && all(prior>0) && all(isfinite(prior)), ...
    'tva:Prior','Policy prior must be positive.');
assert(isequal(size(L),[3 3 3]) && all(L(:)>0) && all(isfinite(L(:))), ...
    'tva:Likelihood','Policy likelihood must be positive 3x3x3.');
assert(numel(costs)==3 && all(costs>=0) && all(isfinite(costs)), ...
    'tva:Costs','Costs must be three nonnegative finite values in nats.');
policy.IG=zeros(1,3); policy.risk=zeros(1,3); policy.ambiguity=zeros(1,3);
policy.joint_risk=zeros(1,3); policy.predicted_observations=zeros(3,3);
policy.preferences=repmat(exp(-costs)/(3*sum(exp(-costs))),3,1);
for a=1:3
    A=L(:,:,a); po=A*prior'; policy.predicted_observations(:,a)=po;
    for c=1:3
        for o=1:3
            joint=prior(c)*A(o,c);
            policy.IG(a)=policy.IG(a)+joint*log(A(o,c)/po(o));
            policy.ambiguity(a)=policy.ambiguity(a)-joint*log(A(o,c));
        end
    end
    policy.risk(a)=sum(po.*log(po/(1/3)));
    policy.joint_risk(a)=sum(po.*log(po./policy.preferences(:,a)));
end
policy.full_efe=policy.joint_risk+policy.ambiguity;
policy.common_constant=log(3)+log(sum(exp(-costs)));
policy.G=costs-policy.IG;
policy.costs=costs; policy.condition=condition;
if strcmp(condition,'no_epistemic'), policy.objective=costs;
else, policy.objective=policy.G; end
if strcmp(condition,'passive'), policy.action_probabilities=[1 0 0];
elseif strcmp(condition,'random_probe'), policy.action_probabilities=[0 .5 .5];
else
    best=abs(policy.objective-min(policy.objective))<1e-12;
    policy.action_probabilities=double(best)/sum(best);
end
end
