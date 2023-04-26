clear all; close all;
addpath ('../');

ax = createAX;
by = createBY;
mem1 = createCTRNNMemory1;
mem12 = createMemory12;
nTimes = 300;

st = '1AAXACAAXX2AAAXCBBABXBY';
ma = length(st);
in = createStrings(st,nTimes);

figure; hold on; grid on;

ax      = resetCTRNN(ax);
by      = resetCTRNN(by);

mem12   = resetCTRNN(mem12);
for i=1:size(in,1)
    mem12.externalInput = in(i,:);
    mem12 = mem12.oneStepUpdate(mem12);
    ax.externalInput = [mem12.outs(end,:) in(i,:)];
    by.externalInput = [mem12.outs(end,:) in(i,:)];
    [ax] = ax.oneStepUpdate (ax);
    [by] = by.oneStepUpdate (by);
end

plotall = 0; 
lw = 4;
out = ax.outs;

plot(out(:,2),'r*','lineWidth',lw)
out = by.outs;
plot(out(:,2),'g*','lineWidth',lw)

out = mem12.outs;
%     plot(out,'g*')
plot(out,'.','lineWidth',lw)
if plotall
    plot(ax.out(:,1),'r*','Color',[0.8,0.1,0.1],'lineWidth',lw)
    plot(bx.out(:,1),'g*','Color',[0.1,0.8,0.1],'lineWidth',lw)
    legend('ax','by','mem1','mem2','in ax','in by')
else
    legend('ax','by','mem1','mem2')
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