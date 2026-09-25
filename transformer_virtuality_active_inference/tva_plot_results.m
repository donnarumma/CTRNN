function paths = tva_plot_results(report, output_dir)
%TVA_PLOT_RESULTS Enumerated task performance and cost-dependent sensing.
if nargin<2, output_dir=fullfile(fileparts(mfilename('fullpath')),'results'); end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
fig=figure('Visible','off','Color','white','Position',[80 80 1200 500]);
guard=onCleanup(@() close(fig));
conditions={'active','random_probe','no_epistemic','fixed_program','oracle','compiled_matched'};
labels={'Active','Random probe','No epistemic term','Frozen program','Oracle','Compiled, matched'};
colors=[.20 .48 .65;.52 .63 .71;.69 .48 .24;.59 .51 .66;.38 .63 .49;.20 .48 .65];
ax1=axes('Parent',fig,'Position',[.15 .17 .32 .72]); hold(ax1,'on');
for i=1:numel(conditions)
    j=find(strcmp({report.summary.condition},conditions{i}));
    accuracy=report.summary(j).accuracy;
    patch([0 accuracy accuracy 0],[i-.32 i-.32 i+.32 i+.32], ...
        colors(i,:),'EdgeColor','none');
    text(accuracy+.015,i,sprintf('%.1f%%',100*accuracy),'FontSize',11, ...
        'HorizontalAlignment','left','VerticalAlignment','middle');
end
set(ax1,'YTick',1:6,'YTickLabel',labels,'YDir','reverse','FontSize',11, ...
    'XTick',[0 1/3 2/3 1],'XTickLabel',{'0','1/3','2/3','1'});
xlim([0 1.14]); ylim([.4 6.6]); set(ax1,'XGrid','on','YGrid','off'); box off;
xlabel('Expected answer accuracy'); title(sprintf('Controls: probe costs %.2f / %.2f', ...
    report.config.costs(2),report.config.costs(3)),'FontSize',12);

ax2=axes('Parent',fig,'Position',[.61 .17 .34 .72]); hold(ax2,'on');
sweep=report.cost_sweep;
active=strcmp({sweep.condition},'active') & [sweep.mode]==1;
ablated=strcmp({sweep.condition},'no_epistemic') & [sweep.mode]==1;
xa=[sweep(active).cost]; ya=[sweep(active).expected_accuracy];
xb=[sweep(ablated).cost]; yb=[sweep(ablated).expected_accuracy];
ha=plot(xa,ya,'o','Color',colors(1,:),'MarkerFaceColor',colors(1,:), ...
    'MarkerSize',6,'LineWidth',1.2);
hb=plot(xb,yb,'s','Color',colors(3,:),'MarkerSize',7,'LineWidth',1.2);
threshold=max([sweep(active).left_IG sweep(active).right_IG]);
plot([threshold threshold],[.25 1],'--','Color',[.50 .35 .61], ...
    'LineWidth',1.2,'HandleVisibility','off');
text(threshold-.025,.98,sprintf('IG = %.3f nats',threshold), ...
    'HorizontalAlignment','right','FontSize',10,'Color',[.45 .26 .57]);
xlim([0 max([sweep.cost])]); ylim([.25 1.02]);
set(ax2,'FontSize',11,'YTick',[1/3 .6 .9 1],'YTickLabel',{'1/3','0.6','0.9','1'});
grid on; box off;
xlabel('Cost of either probe (nats)'); ylabel('Expected answer accuracy');
title('When is sensing worth its cost?','FontSize',12);
legend([ha hb],{'Active','No epistemic term'},'Location','west','Box','off');
set(fig,'PaperUnits','inches','PaperSize',[12 5],'PaperPosition',[0 0 12 5]);
base=fullfile(output_dir,'tva_task_results');
print(fig,[base '.pdf'],'-dpdf'); print(fig,[base '.png'],'-dpng','-r180');
paths=struct('pdf',[base '.pdf'],'png',[base '.png']);
end
