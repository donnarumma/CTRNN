function net = resetCTRNN(net,oneStepUpdate)
% function net = resetCTRNN(net,oneStepUpdate)
    net.indexOut                = 1;
% net.outs(net.indexOut,:)    = net.initialOutValues;
% net.Ys(net.indexOut,:)    = net.initialYValues;
    net.outs    = net.initialOutValues;
    net.Ys      = net.initialYValues;
    if nargin < 2
        net.oneStepUpdate = str2func('ctrnn_dual_fe_oneStep');
    else
        net.oneStepUpdate = str2func(oneStepUpdate);
    end
return