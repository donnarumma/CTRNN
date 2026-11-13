function cpb_plot_results(report, output_dir)
%CPB_PLOT_RESULTS Plot the continual motor-program benchmark from saved data.
%   CPB_PLOT_RESULTS(REPORT, OUTPUT_DIR) creates editable-source, publication
%   figures in PDF and PNG formats. All plotted values are derived from REPORT;
%   no measured result is embedded in this source. Each task row is an exact
%   sensor-outcome mean; this function averages tasks and replications.
%
%   Panels separate end-to-end retention from execution with the correct
%   context supplied. The latter still tests the learned program code: an
%   oracle context does not replace a corrupted or uncertain code.
%
%   Expected REPORT fields: rows, query_rows, config.n_stages.
%   Producing source: cpb_plot_results.m.

if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~isstruct(report) || ~isfield(report, 'rows') || ...
        ~isfield(report, 'query_rows') || ~isfield(report, 'config')
    error('cpb:PlotInput', 'The report must contain rows, query_rows and config.');
end
rows = report.rows;
queries = report.query_rows;
nstages = report.config.n_stages;
stages = 1:nstages;
learner_ids = {'active_archive', 'random_yoked', 'active_fixed', 'random_fixed'};
learner_labels = {'Active / archive', 'Random / yoked budget', ...
                  'Active / fixed budget', 'Random / fixed budget'};
control_ids = {'active_archive', 'compiled_matched', 'dictionary_matched', ...
               'reset_old_codes', 'reset_old_router'};
control_labels = {'Active', 'Compiled', 'Dictionary', 'Code reset', 'Router reset'};
colors = [0.00 0.35 0.60; 0.85 0.33 0.10; 0.25 0.58 0.43; 0.62 0.38 0.69];
styles = {'-', '--', '-', '--'};
markers = {'o', 's', '^', 'd'};

success = nan(nstages, numel(learner_ids));
query_count = nan(nstages, numel(learner_ids));
for k = 1:numel(learner_ids)
    for st = stages
        selected = strcmp({rows.condition}, learner_ids{k}) & [rows.stage] == st;
        success(st, k) = checked_mean(rows, selected, 'success');
        selected = strcmp({queries.condition}, learner_ids{k}) & [queries.stage] == st;
        query_count(st, k) = checked_mean(queries, selected, 'nqueries');
    end
end
final = nan(numel(control_ids), 2);
for k = 1:numel(control_ids)
    selected = strcmp({rows.condition}, control_ids{k}) & [rows.stage] == nstages;
    final(k, 1) = checked_mean(rows, selected, 'success');
    final(k, 2) = checked_mean(rows, selected, 'oracle_context_success');
end
retention = nan(nstages, nstages);
for st = stages
    for acquired = 1:st
        selected = strcmp({rows.condition}, 'active_archive') & ...
                   [rows.stage] == st & [rows.learned_at] == acquired;
        retention(st, acquired) = checked_mean(rows, selected, 'success');
    end
end

if exist('OCTAVE_VERSION', 'builtin')
    toolkits = available_graphics_toolkits();
    if any(strcmp(toolkits, 'gnuplot')), graphics_toolkit('gnuplot'); end
end
f = figure('Visible', 'off', 'Color', 'w', 'Position', [30 30 1280 850]);
if exist('OCTAVE_VERSION', 'builtin') && any(strcmp(toolkits, 'gnuplot'))
    graphics_toolkit(f, 'gnuplot');
end
cleanup = onCleanup(@() close_if_valid(f));
set(f, 'PaperUnits', 'inches', 'PaperPosition', [0 0 12.8 8.5], ...
       'PaperSize', [12.8 8.5]);

ax = subplot(2, 2, 1);
hold(ax, 'on');
for k = 1:numel(learner_ids)
    plot(ax, stages, success(:, k), 'Color', colors(k, :), ...
         'LineStyle', styles{k}, 'Marker', markers{k}, 'LineWidth', 1.7, ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(k, :));
end
format_axes(ax);
xlim(ax, [0.75 nstages + 0.25]); ylim(ax, [0 1.02]);
set(ax, 'XTick', stages, 'YTick', 0:0.2:1);
xlabel(ax, 'Acquisition stage'); ylabel(ax, 'End-to-end success');
title(ax, 'A  Performance on all tasks seen so far', 'FontWeight', 'bold');
ceiling=report.config.cue_reliability+(6-stages)*(1-report.config.cue_reliability)./(5*stages);
plot(ax, stages, ceiling, ':', 'Color', [.25 .25 .25], 'LineWidth', 1.4);
manual_legend(ax, [learner_labels, {'Known-model cue ceiling'}], [colors; .25 .25 .25], ...
    [styles, {':'}], [markers, {'none'}], .06, .39, .072);

ax = subplot(2, 2, 2);
bars = bar(ax, final, 'grouped', 0.80);
set(bars(1), 'FaceColor', colors(1, :), 'EdgeColor', 'none');
set(bars(2), 'FaceColor', [0.67 0.78 0.85], 'EdgeColor', 'none');
format_axes(ax); ylim(ax, [0 1.27]);
set(ax, 'XTick', 1:numel(control_ids), 'XTickLabel', control_labels, ...
        'YTick', 0:0.2:1, 'FontSize', 8.5);
ylabel(ax, 'Final-stage success');
title(ax, 'B  Program integrity and context retrieval', 'FontWeight', 'bold');
manual_legend(ax, {'Inferred context', 'Correct context supplied'}, ...
    [colors(1, :); .67 .78 .85], {'-', '-'}, {'s', 's'}, .06, .93, .075);

ax = subplot(2, 2, 3);
hold(ax, 'on');
for k = 1:numel(learner_ids)
    plot(ax, stages, query_count(:, k), 'Color', colors(k, :), ...
         'LineStyle', styles{k}, 'Marker', markers{k}, 'LineWidth', 1.7, ...
         'MarkerSize', 5, 'MarkerFaceColor', colors(k, :));
end
format_axes(ax);
xlim(ax, [0.75 nstages + 0.25]);
qmax = max(query_count(:));
if isempty(qmax) || ~isfinite(qmax) || qmax <= 0, qmax = 1; end
ylim(ax, [0 qmax * 1.18]); set(ax, 'XTick', stages);
xlabel(ax, 'Acquisition stage'); ylabel(ax, 'Mean acquisition queries');
title(ax, 'C  Information-acquisition budget', 'FontWeight', 'bold');
% Color and marker identities are shared with panel A.

ax = subplot(2, 2, 4);
heat = retention;
heat(isnan(heat)) = 0;
imagesc(ax, stages, stages, heat, [0 1]);
hold(ax, 'on');
% Explicit white cells work on both MATLAB and headless Octave/gnuplot; image
% alpha transparency is not reliably supported by the latter PDF backend.
for st = stages
    for acquired = stages
        if isnan(retention(st, acquired))
            patch(ax, acquired + [-.5 .5 .5 -.5], st + [-.5 -.5 .5 .5], ...
                  'w', 'EdgeColor', [0.92 0.92 0.92]);
        else
            value = retention(st, acquired);
            if value > 0.56, text_color = [1 1 1]; else, text_color = [.1 .1 .1]; end
            text(ax, acquired, st, sprintf('%.2f', value), ...
                 'HorizontalAlignment', 'center', 'Color', text_color, 'FontSize', 10);
        end
    end
end
colormap(f, sequential_map(256));
set(ax, 'XTick', stages, 'YTick', stages, 'YDir', 'reverse', ...
        'TickDir', 'out', 'FontName', 'Helvetica', 'FontSize', 10, 'Box', 'off');
xlim(ax, [.5 nstages + .5]); ylim(ax, [.5 nstages + .5]);
xlabel(ax, 'Stage at which the task was acquired');
ylabel(ax, 'Evaluation stage');
title(ax, 'D  Retention by task age: active archive', 'FontWeight', 'bold');
cb = colorbar(ax);
set(cb, 'YTick', 0:0.25:1, 'FontSize', 9);
ylabel(cb, 'End-to-end success');

% Give the long labels and footer room without relying on tiledlayout.
all_axes=findobj(f, 'Type', 'axes');
set(all_axes, 'Layer', 'top');
for aidx=1:numel(all_axes)
    pos=get(all_axes(aidx),'Position');
    if pos(2)<.2 && pos(4)>.2
        pos(2)=pos(2)+.035; pos(4)=pos(4)-.035;
        set(all_axes(aidx),'Position',pos);
    end
end
annotation(f, 'textbox', [.05 .03 .90 .025], ...
    'String', 'Means across replications; sensor outcomes are averaged exactly.', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Interpreter', 'none', ...
    'FontName', 'Helvetica', 'FontSize', 9, 'Color', [.25 .25 .25]);
annotation(f, 'textbox', [.05 .007 .90 .025], ...
    'String', 'Retention includes context-retrieval competition. White cells: tasks not yet acquired. Source: cpb_plot_results.m.', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Interpreter', 'none', ...
    'FontName', 'Helvetica', 'FontSize', 9, 'Color', [.25 .25 .25]);
drawnow;
print(f, fullfile(output_dir, 'cpb_continual_results.pdf'), '-dpdf', '-painters');
print(f, fullfile(output_dir, 'cpb_continual_results.png'), '-dpng', '-r160');
end

function value = checked_mean(rows, selected, field)
if ~any(selected)
    error('cpb:PlotData', 'No matching observations for plotting field %s.', field);
end
values = [rows(selected).(field)];
if any(~isfinite(values))
    error('cpb:PlotData', 'Nonfinite observations in plotting field %s.', field);
end
value = mean(values);
end

function manual_legend(ax, labels, colors, styles, markers, xfrac, yfrac, step)
% Draw labels explicitly: some Octave/gnuplot versions drop legend objects
% during multiplot PDF/PNG export. These are ordinary data-independent marks.
xl = xlim(ax); yl = ylim(ax); dx = diff(xl); dy = diff(yl);
hold(ax, 'on');
for k = 1:numel(labels)
    y = yl(1) + dy * (yfrac - (k - 1) * step);
    xx = xl(1) + dx * [xfrac, xfrac + .06];
    plot(ax, xx, [y y], 'Color', colors(k, :), 'LineStyle', styles{k}, ...
         'LineWidth', 1.5, 'Marker', markers{k}, 'MarkerSize', 4, ...
         'MarkerFaceColor', colors(k, :));
    text(ax, xl(1) + dx * (xfrac + .08), y, labels{k}, ...
         'FontName', 'Helvetica', 'FontSize', 8.5, ...
         'VerticalAlignment', 'middle', 'Interpreter', 'none');
end
end

function format_axes(ax)
set(ax, 'FontName', 'Helvetica', 'FontSize', 10, 'Box', 'off', ...
        'TickDir', 'out', 'LineWidth', .6);
grid(ax, 'on');
if isprop(ax, 'GridAlpha'), set(ax, 'GridAlpha', .15); end
if isprop(ax, 'GridColor'), set(ax, 'GridColor', [.65 .65 .65]); end
end

function map = sequential_map(n)
% A restrained light-to-dark sequential scale, independent of toolboxes.
anchors = [0.94 .97 .99; .70 .85 .90; .38 .68 .77; .12 .43 .62; .06 .22 .39];
map = interp1(linspace(0, 1, size(anchors, 1)), anchors, linspace(0, 1, n));
end

function close_if_valid(f)
if ishandle(f), close(f); end
end
