function paths = t3_plot_results(report, output_dir)
%T3_PLOT_RESULTS Attention coefficients and exact 3-D convex value mixing.
% report = test_t3_example();
% t3_plot_results(report, fullfile(pwd, 'results'));
% English vector PDF and PNG. Features are coordinates, not neuron positions.
if nargin < 2, output_dir = fullfile(fileparts(mfilename('fullpath')), 'results'); end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
A = report.stage2.A; V = report.stage1.V; Z = report.stage3.Z;
assert(isequal(size(A), [3 3]) && isequal(size(V), [3 3]) && isequal(size(Z), [3 3]));
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [80 80 1200 500]);
cleanup = onCleanup(@() close(fig));

subplot(1, 2, 1);
imagesc(A, [0 1]); axis square;
% A quiet sequential map with enough contrast for black numeric labels.
anchors = [1 1 1; .78 .87 .95; .27 .48 .69];
colormap(interp1([0 .5 1], anchors, linspace(0, 1, 256)));
colorbar;
set(gca, 'XTick', 1:3, 'YTick', 1:3, 'FontSize', 11, 'TickLength', [0 0]);
for i = 1:3
    for j = 1:3
        label_color = [0.08 0.12 0.17];
        if A(i,j) > .8, label_color = [1 1 1]; end
        text(j, i, sprintf('%.4f', A(i,j)), 'HorizontalAlignment', 'center', ...
            'FontSize', 13, 'FontWeight', 'bold', 'Color', label_color);
    end
end
xlabel('Key / value token j'); ylabel('Query / output token i');
title('Attention A (row sums = 1)', 'FontSize', 13);

subplot(1, 2, 2); hold on;
edge = [1 2 3 1];
plot3(V(edge,1), V(edge,2), V(edge,3), '-', 'Color', [.65 .69 .74], ...
    'LineWidth', 1.3, 'HandleVisibility', 'off');
hv = plot3(V(:,1), V(:,2), V(:,3), 'o', 'MarkerSize', 9, ...
    'Color', [.13 .43 .65], 'MarkerFaceColor', [.13 .43 .65]);
hz = plot3(Z(:,1), Z(:,2), Z(:,3), 'd', 'MarkerSize', 8, ...
    'Color', [.49 .27 .67], 'MarkerFaceColor', [.49 .27 .67]);
for i = 1:3
    text(V(i,1), V(i,2), V(i,3)+.09, sprintf('v_%d', i), ...
        'FontSize', 12, 'Color', [.10 .34 .54], 'HorizontalAlignment', 'center');
    text(Z(i,1), Z(i,2), Z(i,3)-.095, sprintf('z_%d', i), ...
        'FontSize', 12, 'Color', [.44 .19 .60], 'HorizontalAlignment', 'center');
end
grid on; box on; axis equal; view(130, 30);
set(gca, 'FontSize', 10, 'XTick', [-.5 0 .5 1], ...
    'YTick', [-.5 0 .5], 'ZTick', [-.5 0 .5 1]);
xlabel(''); ylabel(''); zlabel('Feature 3');
annotation(fig, 'textbox', [.78 .165 .14 .04], 'String', 'Feature 1', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
annotation(fig, 'textbox', [.61 .165 .12 .04], 'String', 'Feature 2', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
title('3D value mixing: Z = A V', 'FontSize', 13);
legend([hv hz], {'Value vectors V', 'Head outputs Z'}, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'Box', 'off');
set(fig, 'PaperUnits', 'inches', 'PaperSize', [12 5], 'PaperPosition', [0 0 12 5]);
base = fullfile(output_dir, 't3_attention_geometry');
print(fig, [base '.pdf'], '-dpdf');
print(fig, [base '.png'], '-dpng', '-r180');
paths = struct('pdf', [base '.pdf'], 'png', [base '.png']);
end
