function model = tva_model(cfg)
% Positive sensor likelihoods, with known reliability mode and hidden selector.
if nargin<1, cfg=struct(); end
if ~isfield(cfg,'sensor_reliabilities'), cfg.sensor_reliabilities=[.9 .55]; end
if ~isfield(cfg,'costs'), cfg.costs=[0 .12 .12]; end
r=cfg.sensor_reliabilities(:)';
assert(numel(r)==2 && all(isfinite(r)) && all(r>0 & r<1), ...
    'tva:Reliability','Supply two sensor diagonal probabilities strictly inside (0,1).');
model.prior=ones(1,3)/3;
model.action_names={'noop','left_probe','right_probe'};
model.mode_names={'left_reliable','right_reliable'};
model.L=zeros(3,3,3,2);
for m=1:2
    model.L(:,:,1,m)=ones(3)/3;
    if m==1, order=[1 2]; else, order=[2 1]; end
    for a=2:3
        reliability=r(order(a-1));
        model.L(:,:,a,m)=reliability*eye(3)+(1-reliability)/2*(ones(3)-eye(3));
    end
end
model.costs=cfg.costs(:)';
model.payload_permutations=perms(1:3);
model.sensor_reliabilities=r;
end
