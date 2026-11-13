function cpb_plot_acquisition(report, output_dir)
%CPB_PLOT_ACQUISITION Compare program learning at identical query counts.
%   CPB_PLOT_ACQUISITION(REPORT, OUTPUT_DIR) reads REPORT.acquisition and
%   exports cpb_acquisition_results.pdf and cpb_acquisition_results.png.
%   Only the two fixed-budget conditions are used. Each plotted prefix
%   includes all acquisition cases, regardless of the learner's confidence;
%   there is no selection on an adaptive stopping time.
%
%   All means, query counts and sample-count labels are derived from REPORT.
%   These are descriptive means across the recorded acquisition cases; no
%   confidence intervals or population generalization claims are produced.
%   Producing source: cpb_plot_acquisition.m.

if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~isstruct(report) || ~isfield(report, 'acquisition')
    error('cpb:AcquisitionPlotInput', 'The report must contain acquisition records.');
end
records = report.acquisition;
ids = {'active_fixed', 'random_fixed'};
labels = {'Active query selection', 'Random query selection'};
colors = [0.00 .35 .60; .85 .33 .10];
styles = {'-', '--'};
markers = {'o', 's'};
steps = unique([records(strcmp({records.condition}, ids{1})).step]);
other_steps = unique([records(strcmp({records.condition}, ids{2})).step]);
if isempty(steps) || ~isequal(steps, other_steps)
    error('cpb:AcquisitionPlotData', 'The two conditions must have identical query prefixes.');
end
steps = reshape(steps, 1, []);
accuracy = zeros(numel(steps), 2);
nll = zeros(numel(steps), 2);
counts = zeros(numel(steps), 2);
initial = records(strcmp({records.condition}, ids{1}) & [records.step] == steps(1));
reference_cases = sortrows([[initial.replication]' [initial.stage]']);
if size(unique(reference_cases, 'rows'), 1) ~= size(reference_cases, 1)
    error('cpb:AcquisitionPlotData', 'Duplicate acquisition cases in a query prefix.');
end
for k = 1:2
    for j = 1:numel(steps)
        selected = strcmp({records.condition}, ids{k}) & [records.step] == steps(j);
        selected_records = records(selected);
        counts(j, k) = numel(selected_records);
        values = [selected_records.code_success];
        losses = [selected_records.code_nll];
        if any(~isfinite(values)) || any(~isfinite(losses)) || ...
                any(values < 0 | values > 1) || any(losses < 0)
            error('cpb:AcquisitionPlotData', 'Invalid acquisition accuracy or log-loss values.');
        end
        accuracy(j, k) = mean(values);
        nll(j, k) = mean(losses);
        compared_cases = sortrows([[selected_records.replication]' [selected_records.stage]']);
        if ~isequal(reference_cases, compared_cases)
            error('cpb:AcquisitionPlotData', 'Unequal acquisition cases at query count %g.', steps(j));
        end
    end
end
if any(counts(:) ~= counts(1))
    error('cpb:AcquisitionPlotData', 'Every prefix must retain the same number of cases.');
end
sample = records(strcmp({records.condition}, ids{1}) & [records.step] == steps(1));
nrep = numel(unique([sample.replication]));
nstage = numel(unique([sample.stage]));

if exist('OCTAVE_VERSION', 'builtin')
    toolkits = available_graphics_toolkits();
    if any(strcmp(toolkits, 'gnuplot')), graphics_toolkit('gnuplot'); end
end
f = figure('Visible', 'off', 'Color', 'w', 'Position', [30 30 1200 520]);
if exist('OCTAVE_VERSION', 'builtin') && any(strcmp(toolkits, 'gnuplot'))
    graphics_toolkit(f, 'gnuplot');
end
cleanup = onCleanup(@() close_if_valid(f));
set(f, 'PaperUnits', 'inches', 'PaperPosition', [0 0 12 5.2], ...
       'PaperSize', [12 5.2]);

ax = axes('Parent', f, 'Position', [.075 .25 .395 .64]);
hold(ax, 'on');
for k = 1:2
    plot(ax, steps, accuracy(:, k), 'Color', colors(k, :), ...
         'LineStyle', styles{k}, 'Marker', markers{k}, 'MarkerSize', 5, ...
         'MarkerFaceColor', colors(k, :), 'LineWidth', 1.7);
end
format_axes(ax);
xlim(ax, [steps(1)-.3 steps(end)+.3]); ylim(ax, [0 1.04]);
set(ax, 'XTick', steps(1):2:steps(end), 'YTick', 0:.2:1);
xlabel(ax, 'Acquisition queries'); ylabel(ax, 'Correct program (MAP decision)');
title(ax, 'A  Program acquisition at equal query budgets', 'FontWeight', 'bold');
manual_legend(ax, labels, colors, styles, markers, .48, .32, .11);

ax = axes('Parent', f, 'Position', [.565 .25 .395 .64]);
hold(ax, 'on');
for k = 1:2
    plot(ax, steps, nll(:, k), 'Color', colors(k, :), ...
         'LineStyle', styles{k}, 'Marker', markers{k}, 'MarkerSize', 5, ...
         'MarkerFaceColor', colors(k, :), 'LineWidth', 1.7);
end
format_axes(ax);
xlim(ax, [steps(1)-.3 steps(end)+.3]);
max_nll = max(nll(:));
if max_nll <= 0, max_nll = 1; end
ylim(ax, [0 max_nll * 1.10]);
set(ax, 'XTick', steps(1):2:steps(end));
xlabel(ax, 'Acquisition queries'); ylabel(ax, 'True-program negative log probability (nats)');
title(ax, 'B  Quality of the learned program belief', 'FontWeight', 'bold');
manual_legend(ax, labels, colors, styles, markers, .42, .87, .11);

count_note = sprintf('Means across %d acquisition cases per query count (%d replications x %d acquisition stages).', ...
    counts(1), nrep, nstage);
annotation(f, 'textbox', [.04 .090 .92 .04], 'String', count_note, ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'Interpreter', 'none', ...
    'FontName', 'Helvetica', 'FontSize', 9, 'Color', [.25 .25 .25]);
annotation(f, 'textbox', [.04 .032 .92 .04], ...
    'String', 'Both conditions retain every case at every prefix. Source: cpb_plot_acquisition.m.', ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'Interpreter', 'none', ...
    'FontName', 'Helvetica', 'FontSize', 9, 'Color', [.25 .25 .25]);
drawnow;
print(f, fullfile(output_dir, 'cpb_acquisition_results.pdf'), '-dpdf', '-painters');
print(f, fullfile(output_dir, 'cpb_acquisition_results.png'), '-dpng', '-r160');
end

function manual_legend(ax, labels, colors, styles, markers, xfrac, yfrac, spacing)
% Explicit marks avoid an Octave/gnuplot legend-export defect.
xl = xlim(ax); yl = ylim(ax); dx = diff(xl); dy = diff(yl);
for k = 1:numel(labels)
    y = yl(1) + dy * (yfrac - (k - 1) * spacing);
    xx = xl(1) + dx * [xfrac xfrac+.06];
    plot(ax, xx, [y y], 'Color', colors(k, :), 'LineStyle', styles{k}, ...
         'Marker', markers{k}, 'MarkerSize', 4, ...
         'MarkerFaceColor', colors(k, :), 'LineWidth', 1.5);
    text(ax, xl(1) + dx * (xfrac+.08), y, labels{k}, ...
         'FontName', 'Helvetica', 'FontSize', 9, 'Interpreter', 'none', ...
         'VerticalAlignment', 'middle');
end
end

function format_axes(ax)
set(ax, 'FontName', 'Helvetica', 'FontSize', 10, 'Box', 'off', ...
        'TickDir', 'out', 'LineWidth', .6);
grid(ax, 'on');
if isprop(ax, 'GridAlpha'), set(ax, 'GridAlpha', .15); end
if isprop(ax, 'GridColor'), set(ax, 'GridColor', [.65 .65 .65]); end
end

function close_if_valid(f)
if ishandle(f), close(f); end
end
