function bci_plot_results(report,output_dir)
% Show raw neural outputs explicitly; normalized KL is only a diagnostic.
if ~exist(output_dir,'dir'), mkdir(output_dir); end
fig=figure('Visible','off','Position',[100 100 1100 790],'Color','w');
guard=onCleanup(@() close(fig));
colors=[.10 .35 .65;.85 .35 .10];
for m=1:2
    ex=report.examples(2*m-1);
    subplot(2,2,m); hold on;
    for i=1:2
        plot(ex.time,ex.target(:,i),'-','Color',colors(i,:),'LineWidth',1.6);
        plot(ex.time,ex.raw(:,i),'--','Color',colors(i,:),'LineWidth',1.6);
    end
    title(sprintf('%s model, same high observation',report.model_names{m}));
    xlabel('Time'); ylabel('Raw neural activity'); ylim([0 1]); grid on;
    legend({'ideal q1','interpreter r1','ideal q2','interpreter r2'},'Location','northeast');
end
subplot(2,2,3);
n=numel(report.config.grid_levels); error_grid=zeros(n,n);
for k=9:numel(report.episodes)
    e=report.episodes(k); index=e.model-4;
    error_grid(index)=max(error_grid(index),e.raw_error);
end
imagesc(report.config.grid_levels,report.config.grid_levels,error_grid');
set(gca,'YDir','normal'); colorbar;
xlabel('P(high | H1)'); ylabel('P(high | H2)');
title('Maximum raw posterior error across the two cues');
subplot(2,2,4); hold on;
ex=report.examples(1);
plot(ex.time,ex.diagnostics.target_kl_curve,'k-','LineWidth',1.5);
plot(ex.time,ex.diagnostics.kl_curve,'--','Color',colors(1,:),'LineWidth',1.5);
xlim([0 min(20,ex.time(end))]); grid on;
xlabel('Time'); ylabel('KL to exact posterior (nat)');
title('Strong model: externally normalized diagnostic');
legend({'ideal circuit','interpreted circuit'},'Location','northeast');
print(fig,fullfile(output_dir,'bayes_circuit_demo.png'),'-dpng','-r150');
end
