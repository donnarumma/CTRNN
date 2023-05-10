close all; clear all;

net = createCTRNNMemory;

figure; hold on; grid on;

net = resetCTRNN(net);
nTimes = 300;

upEq = str2func('ctrnn_dual_fe_oneStep');

for i=1:nTimes
    net.externalInput = [0 0];
    [net] = upEq (net);
end
for i=1:nTimes
    net.externalInput = [1 0];
    [net] = upEq (net);
end
for i=1:nTimes
    net.externalInput = [0 0];
    [net] = upEq (net);
end
for i=1:nTimes
    net.externalInput = [0 1];
    [net] = upEq (net);
end
for i=1:nTimes
    net.externalInput = [0 0];
    [net] = upEq (net);
end

out = net.outs;
% out = net.Ys;
plot(out,'*')
