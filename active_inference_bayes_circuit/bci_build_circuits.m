function bank = bci_build_circuits(models,cfg)
% Two-neuron Bayesian recognition circuits, compiled into ONE fixed interpreter.
% models(m,:) = [P(high|H1), P(high|H2)]. Prior is fixed at [1/2,1/2].
% Code columns are [E11 E12 E21 E22]; physical inputs are [high low] one-hot.
validateattributes(models,{'numeric'},{'2d','real','finite','positive'});
assert(size(models,2)==2 && all(models(:)<1),'bci:Model','Likelihoods must be in (0,1).');
bank.models = models;
bank.prior = [.5 .5];
for m = 1:size(models,1)
    p = models(m,:);
    v = [log(p(1)/p(2)),log((1-p(1))/(1-p(2)))];
    V = [v;-v];
    weights = [V(1,:) V(2,:)];
    assert(all(weights>=cfg.weight_range(1) & weights<=cfg.weight_range(2)), ...
        'bci:Range','Log-likelihood weights exceed the interpreter family.');
    net = createCTRNN(2,2,[0 0],zeros(2),V);
    net.tau = cfg.target_tau*[1 1];
    net.dt = cfg.dt;
    net.initialOutValues = [.5 .5];
    net.externalInput = [1 0];
    net.NOISE = @(n) zeros(1,n.numUnits);
    bank.targets{m} = net;
    bank.weights(m,:) = weights;
    bank.programs(m,:) = weights2program(weights,cfg.weight_range(1),cfg.weight_range(2));
    candidate = net;
    for i = 1:2
        for j = 1:2
            candidate = mulation(candidate,'external',i,j,cfg.weight_range,cfg.multiplier_tau);
        end
    end
    if m==1
        bank.interpreter = candidate;
    else
        fields = {'internalWMatrix','externalWMatrix','biases','tau','initialOutValues', ...
            'numUnits','numExternalInput','dt','outputFun'};
        for f = 1:numel(fields)
            assert(isequal(candidate.(fields{f}),bank.interpreter.(fields{f})), ...
                'bci:Hardware','Program changed physical interpreter field %s.',fields{f});
        end
    end
end
bank.fixed_hardware = true;
assert(bank.interpreter.numUnits==14 && bank.interpreter.numExternalInput==6);
end
