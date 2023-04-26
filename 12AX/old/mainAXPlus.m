clear all; close all;
addpath ('../');

ax = createAX;

nTimes = 300;

st = '1AABACAAXX';
ma = length(st);
in = createStrings(st,nTimes);

figure; hold on; grid on;

ax = resetCTRNN(ax);
for i=1:size(in,1)
    ax.externalInput = in(i,:);
    [ax] = ctrnn_dual_fe_oneStep (ax);
end

out = ax.outs;
plot(out,'*')

legend('ax','delay')
x = [nTimes nTimes];
y = [0,1];

xlim([0,nTimes*ma]);
ylim([0,1]);

for k=1:ma
    plot(k*x,y,'k','lineWidth',2);
end

th = 0.7;
plot([0 ma*nTimes],[th th],'m','lineWidth',2);
th = 0.3;
plot([0 ma*nTimes],[th th],'m','lineWidth',2);

set(gca,'XTick',nTimes/2:nTimes:nTimes*ma)
sss = cell(1,ma);
for k=1:ma
    sss{k}=st(k);
end
set(gca,'XTickLabel',sss)

return