% simHeaviside
net = createCTRNNHeaviside;
figure; hold on; grid on;


net = resetCTRNN(net);
net.externalInput=0;
for i=1:nTimes
    [net] = ctrnn_dual_fe_oneStep (net);
end

out = net.outs;

plot(out,'r*')
% return

net = resetCTRNN(net);
net.externalInput=1;
for i=1:nTimes
    [net] = ctrnn_dual_fe_oneStep (net);
end

out = net.outs;
plot(out,'g*')

legend ('input 0','input 1');
hold off;
return;
