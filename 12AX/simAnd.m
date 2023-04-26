clear all; close all;
and = createCTRNNAnd(3);
nTimes = 200;

figure; hold on; grid on;

% net.externalInput=repmat([1 1],nTimes,1);
% [out]=runCTRNN(net);

and.externalInput = [1,1];
and = resetCTRNN(and);
for i=1:nTimes
    [and] = ctrnn_dual_fe_oneStep (and);
end

out = and.outs;
plot(out,'r*')

% net.externalInput=[net.externalInput; repmat([0 1],nTimes,1)];
and.externalInput=[0 1];
and = resetCTRNN(and);
for i=1:nTimes
    [and] = ctrnn_dual_fe_oneStep (and);
end

out = and.outs;
plot(out,'g*')

% return
and.externalInput=repmat([1 0],nTimes,1);
[out]=runCTRNN(and);

plot(out,'y*')

and.externalInput=repmat([0 0],nTimes,1);
[out]=runCTRNN(and);

plot(out,'b*')

legend ('input 1 1','input 1 0','input 0 1','input 0 0');
hold off;
return;