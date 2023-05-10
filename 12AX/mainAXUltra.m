clear all; close all;
addpath ('../');

ax      = createAX;
by      = createBY;

mem12   = createMemory12;
prog    = createProgramSend;

nTimes  = 300;

st = '1AAAXACAAXX2AAAXCBBABXBYY1BB';
ma = length(st);
in = createStrings(st,nTimes);

figure; hold on; grid on;

ifmul = 1;

plotall = 0; 

if ifmul
    % substitutes the different weights 
    mi = [-2 5];
    mt = 0.8;
    ax=mulation(ax,'external',1,1,mi,mt);
    ax=mulation(ax,'external',1,2,mi,mt);
    ax=mulation(ax,'external',1,4,mi,mt);
    ax=mulation(ax,'external',2,5,mi,mt);
    ax=mulation(ax,'external',1,7,mi,mt);
    ax=mulation(ax,'external',2,8,mi,mt);
    
    by=mulation(by,'external',1,1,mi,mt);
    by=mulation(by,'external',1,2,mi,mt);
    by=mulation(by,'external',1,4,mi,mt);
    by=mulation(by,'external',2,5,mi,mt);
    by=mulation(by,'external',1,7,mi,mt);
    by=mulation(by,'external',2,8,mi,mt);
    
end

ax      = resetCTRNN(ax);
by      = resetCTRNN(by);
mem12   = resetCTRNN(mem12);
prog    = resetCTRNN(prog);

so = mem12.numUnits + size(in,2);
pax = ax.externalInput(:,so+1:end);
pby = by.externalInput(:,so+1:end);

for i=1:size(in,1)
    mem12.externalInput = in(i,:);
    mem12   = mem12.oneStepUpdate(mem12);
    prog.externalInput = mem12.outs(end,:);
    prog    = prog.oneStepUpdate(prog);
    
    if ~ifmul
        ax.externalInput = [mem12.outs(end,:) in(i,:) pax];
        by.externalInput = [mem12.outs(end,:) in(i,:) pby];
        [ax] = ax.oneStepUpdate (ax);
        [by] = by.oneStepUpdate (by);
    else
        ax.externalInput = [mem12.outs(end,:) in(i,:) prog.outs(end,:)];
        [ax] = ax.oneStepUpdate (ax);
    end
end

lw = 4;
out = ax.outs;

plot(out(:,2),'r*','lineWidth',lw)
if ~ifmul
    out = by.outs;
    plot(out(:,2),'g*','lineWidth',lw)
end

out = mem12.outs;
%     plot(out,'g*')
plot(out,'.','lineWidth',lw)
if plotall
    plot(ax.outs(:,1),'r*','Color',[0.8,0.1,0.1],'lineWidth',lw)
    plot(by.outs(:,1),'g*','Color',[0.1,0.8,0.1],'lineWidth',lw)
    if ifmul
        legend('InterpreterAXBY','mem1','mem2','in ax','in by')
    else
        legend('ax','by','mem1','mem2','in ax','in by')
    end
else
    if ifmul
        legend('InterpreterAXBY','mem1','mem2')
    else
        legend('ax','by','mem1','mem2')
    end
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