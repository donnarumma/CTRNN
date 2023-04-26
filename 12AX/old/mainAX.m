clear all; close all;
ax = createAX;
nTimes = 300;%(1/and.dt)*and.tau*3;

figure; hold on; grid on;

%1
%A
ax.externalInput = [1,1,0,0,0,0,0,0];
ax = resetCTRNN(ax);
for i=1:nTimes
    [ax] = ctrnn_dual_fe_oneStep (ax);
end

%X
ax.externalInput=[1,0,1,0,0,0,0,0];

for i=1:nTimes
    [ax] = ctrnn_dual_fe_oneStep (ax);
end

%X
ax.externalInput=[1,0,1,0,0,0,0,0];

for i=1:nTimes
    [ax] = ctrnn_dual_fe_oneStep (ax);
end


out = ax.outs;
plot(out,'*')
% out = not.outs;
% plot(out,'r*')
legend('ax','delay')
x = [nTimes nTimes];
y = [0,1];

ma=3;

xlim([0,nTimes*ma]);
ylim([0,1]);

for k=1:ma
    plot(k*x,y,'k');
end
th = 0.7;
plot([0 ma*nTimes],[th th],'m','lineWidth',2);
th = 0.3;
plot([0 ma*nTimes],[th th],'m','lineWidth',2);


%  [a b]=min(abs(not.outs(nTimes+1:end,:)-0.5));
%  fprintf('tt=%g\n',b/(nTimes));
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