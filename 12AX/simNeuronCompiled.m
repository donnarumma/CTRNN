% simNeuronCompiled
clear all; close all;
addpath ('../');
load progs;
nTimes = 300;

figure; hold on; grid on;

% net     = createNeuronCompiled;
net     = createProgramSend;
net     = resetCTRNN(net);


for i=1:nTimes
    net.externalInput = [1 0];
    net = net.oneStepUpdate (net);
end

out = net.outs;
p0 = pax(1);
out(end)-p0


for i=1:nTimes
    net.externalInput = [0 1];
    net = net.oneStepUpdate (net);
end

lw = 4;

p0 = pax(2);
out = net.outs;
out(end)-p0

plot(out,'r*','lineWidth',lw);

hold off;    

