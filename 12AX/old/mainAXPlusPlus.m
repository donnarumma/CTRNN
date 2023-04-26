clear all; close all;
addpath ('../');

ax = createAX;
mem1 = createCTRNNMemory1;

nTimes = 300;

st = '1AAXACAAXX2AAAXC';
ma = length(st);
in = createStrings(st,nTimes);

figure; hold on; grid on;

ax      = resetCTRNN(ax);
mem1    = resetCTRNN(mem1);
for i=1:size(in,1)
    mem1.externalInput = in(i,:);
    mem1 = mem1.oneStepUpdate(mem1);
    ax.externalInput = [mem1.outs(end,:) in(i,:)];
    [ax] = ax.oneStepUpdate (ax);
end

plotall = 0; 
if plotall
    out = ax.outs;
    plot(out,'*')

    out = mem1.outs;
    plot(out,'*')
    legend('ax','delay','mem1')
else
    out = ax.outs;
    plot(out(:,2),'b*')

    out = mem1.outs;
    plot(out,'g*')
    legend('ax','mem1')
end
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