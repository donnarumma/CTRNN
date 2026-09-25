function paths = mpb_plot_results(report, output_dir)
%MPB_PLOT_RESULTS Plot exact benchmark expectations and motor trajectories.
% All plotted quantities come from REPORT; no sampling error bars are added
% to exactly enumerated outcomes. Figure sources use only MATLAB/Octave APIs.
if nargin < 2
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
required = {'summary', 'cost_sweep', 'examples', 'protocol'};
for i = 1:numel(required)
    if ~isfield(report, required{i})
        error('mpb:plot:MissingField', 'Report is missing %s.', required{i});
    end
end

active_color = [.16 .43 .64];
noepi_color = [.77 .39 .13];
nolook_color = [.40 .43 .47];
reliable_color = [.24 .57 .39];
conditions = {'active','no_epistemic','no_look','matched_random_look', ...
    'compiled_matched','oracle','frozen_motor_keys', ...
    'frozen_motor_attention','reset_memory'};
labels = {'Active','No epistemic term','No LOOK','Random LOOK (pooled budget)', ...
    'Compiled, matched','Oracle','Frozen motor keys', ...
    'Frozen motor attention','Reset memory'};
colors = [active_color; noepi_color; nolook_color; .25 .58 .59; ...
    active_color; reliable_color; .57 .43 .66; .71 .37 .49; .50 .25 .29];

fig = figure('Visible','off','Color','white', ...
    'Position',[60 60 1400 900]);
guard = onCleanup(@() close(fig));

ax = axes('Parent',fig,'Position',[.16 .59 .29 .32]);
hold(ax,'on');
for i = 1:numel(conditions)
    value = summary_value(report, 'evaluation', 'mixture', ...
        conditions{i}, 'success');
    patch([0 value value 0], [i-.33 i-.33 i+.33 i+.33], colors(i,:), ...
        'EdgeColor','none','Parent',ax);
    text(value+.017, i, sprintf('%.1f%%',100*value), 'Parent',ax, ...
        'FontSize',10,'HorizontalAlignment','left', ...
        'VerticalAlignment','middle');
end
set(ax,'FontSize',10,'YTick',1:numel(conditions),'YTickLabel',labels, ...
    'YDir','reverse','XTick',0:.25:1,'XGrid','on','YGrid','off');
xlim(ax,[0 1.17]); ylim(ax,[.35 numel(conditions)+.65]); box(ax,'off');
xlabel(ax,'Exact probability of ordered completion');
title(ax,'A. Controls: regime mixture, evaluation layouts','FontSize',11);

ax = axes('Parent',fig,'Position',[.60 .59 .35 .32]);
hold(ax,'on');
regimes = report.protocol.regime_names;
reliability = report.protocol.initial_reliability;
belief_conditions = {'active','no_epistemic','no_look'};
belief_labels = {'Active','No epistemic term','No LOOK'};
belief_colors = [active_color;noepi_color;nolook_color];
markers = {'o','s','^'};
styles = {'-','--',':'};
handles = cell(1,numel(belief_conditions));
for i = 1:numel(belief_conditions)
    vals = zeros(size(reliability));
    for j = 1:numel(regimes)
        vals(j) = summary_value(report, 'evaluation', regimes{j}, ...
            belief_conditions{i}, 'decision_nll');
    end
    handles{i} = plot(ax,reliability,vals,'Color',belief_colors(i,:), ...
        'LineStyle',styles{i},'Marker',markers{i}, ...
        'MarkerSize',7,'LineWidth',1.6);
end
set(ax,'FontSize',10,'XTick',reliability); grid(ax,'on'); box(ax,'off');
xlim(ax,[max(0,min(reliability)-.04) min(1,max(reliability)+.04)]);
xlabel(ax,'Initial cue reliability');
ylabel(ax,'Expected posterior log loss (nats)');
title(ax,'B. Belief quality at the route decision','FontSize',11);
legend(ax,[handles{:}],belief_labels,'Location','northeast','Box','off');

ax = axes('Parent',fig,'Position',[.10 .12 .35 .32]);
hold(ax,'on');
sensing_regimes = {'intermediate','reliable'};
sensing_colors = [active_color;reliable_color];
sensing_conditions = {'active','no_epistemic'};
sensing_names = {'Intermediate: active','Intermediate: no epistemic term', ...
    'Reliable: active','Reliable: no epistemic term'};
handles = cell(1,4); cursor = 0;
for r = 1:numel(sensing_regimes)
    for c = 1:numel(sensing_conditions)
        [cost, vals] = sweep_values(report, sensing_regimes{r}, ...
            sensing_conditions{c}, 'looks');
        cursor = cursor+1;
        handles{cursor} = plot(ax,cost,vals,'Color',sensing_colors(r,:), ...
            'LineStyle',styles{c},'Marker',markers{c}, ...
            'MarkerSize',5,'LineWidth',1.4);
    end
end
set(ax,'FontSize',10); grid(ax,'on'); box(ax,'off'); ylim(ax,[-.04 1.05]);
xlabel(ax,'Cost of LOOK (nats)'); ylabel(ax,'Probability of taking LOOK');
title(ax,'C. Sensing depends on cost and uncertainty','FontSize',11);
legend(ax,[handles{:}],sensing_names,'Location','east','Box','off', ...
    'FontSize',9);

ax = axes('Parent',fig,'Position',[.60 .12 .35 .32]);
hold(ax,'on');
handles = cell(1,2);
for i = 1:2
    [~,execution_cost] = sweep_values(report,'intermediate', ...
        sensing_conditions{i},'total_cost');
    [~,success] = sweep_values(report,'intermediate', ...
        sensing_conditions{i},'success');
    handles{i} = plot(ax,execution_cost,success, ...
        'Color',belief_colors(i,:),'LineStyle',styles{i}, ...
        'Marker',markers{i},'MarkerSize',7,'LineWidth',1.6);
end
set(ax,'FontSize',10); grid(ax,'on'); box(ax,'off'); ylim(ax,[-.02 1.05]);
xlabel(ax,'Expected execution + sensing cost (nats)');
ylabel(ax,'Probability of ordered completion');
title(ax,'D. Outcomes across LOOK costs: intermediate cue','FontSize',11);
legend(ax,[handles{:}],{'Active','No epistemic term'}, ...
    'Location','southeast','Box','off');

base = fullfile(output_dir,'mpb_benchmark_results');
save_figure(fig,base,[14 9]);
paths = struct('results_pdf',[base '.pdf'],'results_png',[base '.png']);
clear guard;

example_labels = {'interpreted_correct','frozen_keys','reset_memory'};
example_titles = {'A. Interpreted execution','B. Frozen motor keys', ...
    'C. Reset progress memory'};
fig = figure('Visible','off','Color','white', ...
    'Position',[60 60 1400 500]);
guard = onCleanup(@() close(fig));
for i = 1:numel(example_labels)
    match = find(strcmp({report.examples.label},example_labels{i}));
    if numel(match) ~= 1
        error('mpb:plot:Example', 'Expected one example named %s.', ...
            example_labels{i});
    end
    example = report.examples(match);
    ax = axes('Parent',fig,'Position',[.05+(i-1)*.325 .26 .275 .59]);
    draw_example(ax, example, example_titles{i}, active_color);
end
base = fullfile(output_dir,'mpb_motor_example');
save_figure(fig,base,[14 5]);
paths.motor_pdf = [base '.pdf']; paths.motor_png = [base '.png'];
end

function idx = metric_index(report,name)
idx = find(strcmp(report.protocol.metrics,name));
if numel(idx) ~= 1
    error('mpb:plot:Metric','Missing or duplicate metric %s.',name);
end
end

function value = summary_value(report,split,regime,condition,metric)
rows = report.summary;
idx = find(strcmp({rows.split},split) & strcmp({rows.regime},regime) & ...
    strcmp({rows.condition},condition));
if numel(idx) ~= 1
    error('mpb:plot:Summary', 'Expected one %s/%s/%s summary row.', ...
        split,regime,condition);
end
value = rows(idx).values(metric_index(report,metric));
end

function [cost,values] = sweep_values(report,regime,condition,metric)
rows = report.cost_sweep;
idx = find(strcmp({rows.regime},regime) & ...
    strcmp({rows.condition},condition));
if isempty(idx)
    error('mpb:plot:Sweep','No cost sweep for %s/%s.',regime,condition);
end
cost = [rows(idx).look_cost]; [cost,order] = sort(cost);
idx = idx(order); values = zeros(size(cost));
mi = metric_index(report,metric);
for j = 1:numel(idx), values(j) = rows(idx(j)).values(mi); end
end

function save_figure(fig,base,dimensions)
set(fig,'PaperUnits','inches','PaperSize',dimensions, ...
    'PaperPosition',[0 0 dimensions]);
print(fig,[base '.pdf'],'-dpdf');
print(fig,[base '.png'],'-dpng','-r180');
end

function draw_example(ax,example,panel_title,path_color)
layout = example.layout; trace = example.trace;
n = layout.grid_size;
if numel(n) == 1, n = [n n]; end
hold(ax,'on');
for row = .5:1:n(1)+.5
    plot(ax,[.5 n(2)+.5],[row row],'-','Color',[.86 .86 .86], ...
        'LineWidth',.6,'HandleVisibility','off');
end
for col = .5:1:n(2)+.5
    plot(ax,[col col],[.5 n(1)+.5],'-','Color',[.86 .86 .86], ...
        'LineWidth',.6,'HandleVisibility','off');
end
draw_obstacles(ax,layout,n);
if ~isempty(trace)
    route = [trace(1,2:3);trace(:,4:5)];
    plot(ax,route(:,2),route(:,1),'-','Color',path_color, ...
        'LineWidth',2.2,'HandleVisibility','off');
    delta = diff(route,1,1); moving = any(delta~=0,2);
    origin = route(1:end-1,:);
    for step = find(moving)'
        v = [delta(step,2),delta(step,1)]; v = v/norm(v);
        center = [origin(step,2),origin(step,1)] + ...
            .5*[delta(step,2),delta(step,1)];
        side = [-v(2),v(1)];
        triangle = [center+.13*v; center-.10*v+.07*side; ...
            center-.10*v-.07*side];
        patch(triangle(:,1),triangle(:,2),path_color, ...
            'EdgeColor','none','Parent',ax,'HandleVisibility','off');
    end
end
goal_names = 'ABCD';
goal_colors = [.87 .74 .29;.41 .69 .46;.65 .49 .72;.91 .54 .33];
for g = 1:size(layout.landmarks,1)
    point = layout.landmarks(g,:);
    plot(ax,point(2),point(1),'o','MarkerSize',20, ...
        'MarkerFaceColor',goal_colors(g,:),'MarkerEdgeColor',[.3 .3 .3], ...
        'HandleVisibility','off');
    text(point(2),point(1),goal_names(g),'Parent',ax, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',12,'FontWeight','bold');
end
plot(ax,layout.start(2),layout.start(1),'s','MarkerSize',17, ...
    'MarkerFaceColor',[.97 .97 .97],'MarkerEdgeColor',[.2 .2 .2], ...
    'LineWidth',1.2,'HandleVisibility','off');
text(layout.start(2),layout.start(1),'S','Parent',ax, ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'FontSize',11,'FontWeight','bold');
if ~isempty(trace)
    failed = trace(:,11) > 0;
    if any(failed)
        plot(ax,trace(failed,5),trace(failed,4),'x', ...
            'Color',[.75 .10 .12],'MarkerSize',26,'LineWidth',2.4, ...
            'HandleVisibility','off');
    end
end
set(ax,'YDir','reverse','FontSize',10,'XTick',1:n(2),'YTick',1:n(1), ...
    'DataAspectRatio',[1 1 1]);
xlim(ax,[.5 n(2)+.5]); ylim(ax,[.5 n(1)+.5]); box(ax,'on');
title(ax,panel_title,'FontSize',11,'FontWeight','normal');
xlabel(ax,'Column','FontSize',10);
text(.5,-.22,['Diagnostic: clamped program ' route_text(example.program)], ...
    'Parent',ax,'Units','normalized','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontSize',10,'Interpreter','none');
text(.5,-.32,['Required: ' route_text(example.truth) ...
    '; red cross = wrong goal'], ...
    'Parent',ax,'Units','normalized','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontSize',9,'Interpreter','none');
ylabel(ax,'Row');
end

function str = route_text(program)
names = 'ABCD';
if any(program < 1) || any(program > numel(names)) || ...
        any(program ~= floor(program))
    error('mpb:plot:Program','Expected goal indices 1 through 4.');
end
str = names(program(1));
for k = 2:numel(program), str = [str ' > ' names(program(k))]; end
end

function draw_obstacles(ax,layout,n)
obstacles = [];
if isfield(layout,'blocked'), obstacles = layout.blocked;
elseif isfield(layout,'walls'), obstacles = layout.walls;
elseif isfield(layout,'obstacles'), obstacles = layout.obstacles;
end
if isempty(obstacles), return; end
if isequal(size(obstacles),n)
    [r,c] = find(obstacles); obstacles = [r c];
end
if size(obstacles,2) ~= 2
    error('mpb:plot:Obstacles','Obstacles must be a grid mask or N-by-2 coordinates.');
end
for i = 1:size(obstacles,1)
    row = obstacles(i,1); col = obstacles(i,2);
    patch(col+[-.5 .5 .5 -.5],row+[-.5 -.5 .5 .5], ...
        [.35 .37 .40],'EdgeColor','none','Parent',ax);
end
end
