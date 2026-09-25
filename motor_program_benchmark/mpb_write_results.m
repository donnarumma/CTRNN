function mpb_write_results(report,output_dir)
%MPB_WRITE_RESULTS Auditable numeric exports; no graphics in this function.
if ~exist(output_dir,'dir'), mkdir(output_dir); end
save(fullfile(output_dir,'mpb_report.mat'),'report','-v7');
write_json(fullfile(output_dir,'protocol.json'),report.protocol);
summary_export=struct('metadata',report.metadata,'summary',report.summary, ...
    'validation',report.validation,'core_stats',report.core_stats,'structural_counts',report.structural_counts);
if isfield(report,'spm'), summary_export.spm=rmfield(report.spm,'percheck'); end
write_json(fullfile(output_dir,'summary.json'),summary_export);
metrics=report.protocol.metrics;
rows=cell(numel(report.summary),4+numel(metrics));
for k=1:numel(report.summary)
    s=report.summary(k); rows(k,:)=[{s.split,s.regime,s.condition,s.n_layouts},num2cell(s.values)];
end
write_csv(fullfile(output_dir,'summary.csv'),[{'split','regime','condition','n_layouts'},metrics],rows);
rows=cell(numel(report.layout_metrics),5+numel(metrics));
for k=1:numel(report.layout_metrics)
    s=report.layout_metrics(k); rows(k,:)=[{s.layout,s.split,s.regime,s.condition,s.mass},num2cell(s.values)];
end
write_csv(fullfile(output_dir,'layout_metrics.csv'),[{'layout','split','regime','condition','probability_mass'},metrics],rows);
rows=cell(numel(report.route_metrics),6+numel(metrics));
for k=1:numel(report.route_metrics)
    s=report.route_metrics(k); rows(k,:)=[{s.layout,s.split,s.regime,s.condition,s.route,s.composition},num2cell(s.values)];
end
write_csv(fullfile(output_dir,'route_metrics.csv'),[{'layout','split','regime','condition','route','composition'},metrics],rows);
rows=cell(numel(report.cost_sweep),4+numel(metrics));
for k=1:numel(report.cost_sweep)
    s=report.cost_sweep(k); rows(k,:)=[{s.regime,s.condition,s.look_cost,s.n_layouts},num2cell(s.values)];
end
write_csv(fullfile(output_dir,'cost_sweep.csv'),[{'regime','condition','look_cost','n_layouts'},metrics],rows);
rows=cell(numel(report.layouts),14);
for k=1:numel(report.layouts)
    s=report.layouts(k); coords=reshape(s.landmarks',1,8);
    rows(k,:)=[{s.id,s.split,s.seed,s.grid_size},num2cell([s.start,coords])];
end
write_csv(fullfile(output_dir,'layouts.csv'),{'layout','split','seed','grid_size','start_row','start_col', ...
    'A_row','A_col','B_row','B_col','C_row','C_col','D_row','D_col'},rows);
rows=cell(numel(report.belief_checks),24);
headers={'regime','initial_cue','probe_cue'};
for prefix={'prior','posterior','reference'}
    for j=1:6, headers{end+1}=sprintf('%s_%d',prefix{1},j); end %#ok<AGROW>
end
headers=[headers,{'max_error','F_before','F_after'}];
for k=1:numel(report.belief_checks)
    s=report.belief_checks(k); rows(k,:)=[{s.regime,s.initial_cue,s.probe_cue}, ...
        num2cell([s.prior,s.posterior,s.reference,s.max_error,s.F_before,s.F_after])];
end
write_csv(fullfile(output_dir,'belief_checks.csv'),headers,rows);
trace_headers={'time','before_row','before_col','after_row','after_col','stage','action', ...
    'touched_goal','accepted_count','success','wrong_goal','program_id','true_route','action_confidence'};
for k=1:numel(report.examples)
    s=report.examples(k); write_csv(fullfile(output_dir,['trace_' s.label '.csv']),trace_headers,num2cell(s.trace));
end
fid=fopen(fullfile(output_dir,'mpb_summary_rows.tex'),'w'); assert(fid>=0); closefid=onCleanup(@()fclose(fid)); %#ok<NASGU>
labels={'Active interpreter','No epistemic term','No additional look','Random look, matched budget', ...
    'Compiled, matched information','Oracle rule','Frozen motor keys','Frozen motor attention','Reset phase memory'};
for k=1:numel(report.protocol.conditions)
    cond=report.protocol.conditions{k};
    match=strcmp({report.summary.split},'evaluation') & strcmp({report.summary.regime},'mixture') & strcmp({report.summary.condition},cond);
    v=report.summary(match).values;
    fprintf(fid,'%s & %.1f & %.2f & %.3f & %.3f %s%c', ...
        labels{k},100*v(1),v(6),v(8),v(9),char([92 92]),10);
end
clear closefid
fid=fopen(fullfile(output_dir,'mpb_metrics.tex'),'w'); assert(fid>=0); closemetrics=onCleanup(@()fclose(fid)); %#ok<NASGU>
write_macro(fid,'MpbBayesError',tex_scientific(report.core_stats.max_bayes_error));
write_macro(fid,'MpbCompiledError',tex_scientific(report.core_stats.max_compiled_error));
if isfield(report,'spm')
    write_macro(fid,'MpbSpmError',tex_scientific(report.spm.maxerror));
    write_macro(fid,'MpbSpmChecks',sprintf('%d',report.spm.nchecks));
else
    write_macro(fid,'MpbSpmError','\mathrm{not\ checked}');
    write_macro(fid,'MpbSpmChecks','0');
end
end

function write_macro(fid,name,value)
fprintf(fid,'%s%c',[char(92) 'newcommand{' char(92) name '}{' value '}'],10);
end

function value=tex_scientific(x)
if x==0, value='0'; return; end
exponent=floor(log10(abs(x))); mantissa=x/10^exponent;
value=[sprintf('%.3f',mantissa) char(92) 'times10^{' sprintf('%d',exponent) '}'];
end

function write_json(path,data)
fid=fopen(path,'w'); assert(fid>=0); closer=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(data));
end

function write_csv(path,headers,rows)
fid=fopen(path,'w'); assert(fid>=0); closer=onCleanup(@()fclose(fid)); %#ok<NASGU>
assert(size(rows,2)==numel(headers),'mpb:Export','CSV width mismatch.');
fprintf(fid,'%s\n',strjoin(headers,','));
for r=1:size(rows,1)
    vals=cell(1,size(rows,2));
    for col=1:size(rows,2)
        value=rows{r,col};
        if isnumeric(value), vals{col}=sprintf('%.17g',value);
        else, vals{col}=['"' strrep(value,'"','""') '"']; end
    end
    fprintf(fid,'%s\n',strjoin(vals,','));
end
end
