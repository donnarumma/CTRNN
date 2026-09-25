function ai2_plot_results(report, output_dir)
% Static figure for the report and LaTeX notes. No interactive UI required.
if ~exist(output_dir,'dir'), mkdir(output_dir); end
fig = figure('Visible','off','Position',[100 100 1100 760],'Color','w');
guard = onCleanup(@() close(fig));
colors = [0.10 .35 .65; .85 .35 .10];
for context = 1:2
    subplot(2,2,context);
    hold on;
    for neuron = 1:2
        plot(report.examples(context).time,report.examples(context).target(:,neuron), ...
            '-','Color',colors(neuron,:),'LineWidth',1.5);
        plot(report.examples(context).time,report.examples(context).output(:,neuron), ...
            '--','Color',colors(neuron,:),'LineWidth',1.5);
    end
    xlabel('Time'); ylabel('Neural activity'); ylim([0 1]); grid on;
    title(sprintf('Program %d: target vs interpreter',context));
    legend({'target n1','interpreter n1','target n2','interpreter n2'},'Location','best');
end
subplot(2,2,3);
bar([[report.summary.accuracy]' [report.summary.informative_probe_rate]']);
set(gca,'XTick',1:3,'XTickLabel',{report.summary.condition});
ylim([0 1.08]); grid on;
legend({'correct program','informative probe'},'Location','best');
title(sprintf('%d episodes / condition; seed %d',report.summary(1).n,report.config.seed));
subplot(2,2,4);
imagesc(report.programs,[0 1]); colorbar;
set(gca,'XTick',1:4,'XTickLabel',{'W11','W22','V11','V22'}, ...
    'YTick',1:2,'YTickLabel',{'program 1','program 2'});
title('Codes supplied to the same fixed interpreter');
for row = 1:2
    for col = 1:4
        text(col,row,sprintf('%.2f',report.programs(row,col)), ...
            'HorizontalAlignment','center','Color','k','FontWeight','bold');
    end
end
print(fig,fullfile(output_dir,'two_neuron_demo.png'),'-dpng','-r150');
end
