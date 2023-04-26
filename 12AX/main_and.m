clear all; close all;
and = createCTRNNAnd(9);
not = createCTRNNHeavi(3);%createCTRNNNot(1);
nTimes = 300;%(1/and.dt)*and.tau*3;

figure; hold on; grid on;

% net.externalInput=repmat([1 1],nTimes,1);
% [out]=runCTRNN(net);

and.externalInput = [1,1];
and = resetCTRNN(and);
for i=1:nTimes
    
    [and] = ctrnn_dual_fe_oneStep (and);
    
    not.externalInput = [and.outs(end,:) 0];
    [not] = ctrnn_dual_fe_oneStep (not);
    
end



% net.externalInput=[net.externalInput; repmat([0 1],nTimes,1)];
and.externalInput=[0 1];
% and = resetCTRNN(and);
for i=1:nTimes
    [and] = ctrnn_dual_fe_oneStep (and);
    
    not.externalInput = [and.outs(end,:) 1];
    [not] = ctrnn_dual_fe_oneStep (not);
end

and.externalInput=[0 1];
% and = resetCTRNN(and);
for i=1:nTimes
    [and] = ctrnn_dual_fe_oneStep (and);
    
    not.externalInput = [and.outs(end,:) 1];
    [not] = ctrnn_dual_fe_oneStep (not);
end


out = and.outs;
plot(out,'g*')
out = not.outs;
plot(out,'r*')
legend('and','delay')
 [a b]=min(abs(not.outs(nTimes+1:end,:)-0.5));
 fprintf('tt=%g\n',b/(nTimes));
return

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