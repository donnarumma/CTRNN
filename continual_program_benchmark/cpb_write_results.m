function cpb_write_results(report,output_dir)
%CPB_WRITE_RESULTS Export complete report and numeric tables with honest labels.
if ~exist(output_dir,'dir'), mkdir(output_dir); end
save(fullfile(output_dir,'cpb_report.mat'),'report','-v7');
write_json(fullfile(output_dir,'protocol.json'),report.config);
summary_export=struct('metadata',report.metadata, ...
 'summary',report.summary,'validation',report.validation,'core_tests',report.core_tests, ...
 'motor_validation',rmfield(report.motor,{'layouts','cost','success'}));
summary_export.integration_tests=report.integration_tests;
if isfield(report,'spm'), summary_export.spm=report.spm; end
write_json(fullfile(output_dir,'summary.json'),summary_export);
write_struct_csv(fullfile(output_dir,'retention.csv'),report.rows);
write_struct_csv(fullfile(output_dir,'queries.csv'),report.query_rows);
write_struct_csv(fullfile(output_dir,'acquisition.csv'),report.acquisition);
write_struct_csv(fullfile(output_dir,'summary.csv'),report.summary);
write_struct_csv(fullfile(output_dir,'learning_streams.csv'),stream_rows(report));
labels={'Active archive','Random queries, yoked count','Active, fixed budget','Random, fixed budget', ...
 'Compiled, matched','Dictionary, matched','Reset old program beliefs','Reset old cue model'};
fid=fopen(fullfile(output_dir,'cpb_summary_rows.tex'),'w'); assert(fid>=0); guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
for k=1:numel(report.summary)
 s=report.summary(k); q=report.query_rows(strcmp({report.query_rows.condition},s.condition));
 if isempty(q), q=report.query_rows(strcmp({report.query_rows.condition},'active_archive')); end
 fprintf(fid,'%s & %.1f & %.1f & %.2f %s%c',labels{k},100*s.success, ...
 100*s.oracle_context_success,mean([q.nqueries]),char([92 92]),10);
end
clear guard
fid=fopen(fullfile(output_dir,'cpb_metrics.tex'),'w'); assert(fid>=0); guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
write_macro(fid,'CpbReplications',sprintf('%d',report.config.n_replications));
write_macro(fid,'CpbLayouts',sprintf('%d',report.motor.n_layouts));
write_macro(fid,'CpbRollouts',sprintf('%d',report.motor.n_rollouts));
write_macro(fid,'CpbMotorError',sprintf('%.3g',report.motor.max_compiled_command_error));
write_macro(fid,'CpbCodeDrift',sprintf('%.3g',report.validation.max_stored_code_drift));
end
function rows=stream_rows(report)
f=report.fixture; rows=struct('replication',{},'stage',{},'task',{},'true_code',{},'cue_peak',{});
for r=1:report.config.n_replications
 for b=1:6
  task=f.order(r,b);
  rows(end+1)=struct('replication',r,'stage',b,'task',task,'true_code',f.task_code(r,task), ...
    'cue_peak',f.task_cue(r,task)); %#ok<AGROW>
 end
end
end
function write_json(path,data)
fid=fopen(path,'w'); assert(fid>=0); guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s%c',jsonencode(data),10);
end
function write_struct_csv(path,rows)
fid=fopen(path,'w'); assert(fid>=0); guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
fields=fieldnames(rows); fprintf(fid,'%s%c',strjoin(fields',','),10);
for r=1:numel(rows)
 values=cell(1,numel(fields));
 for k=1:numel(fields)
  x=rows(r).(fields{k});
  if isnumeric(x) || islogical(x), values{k}=sprintf('%.17g',x);
  else, values{k}=['"' strrep(x,'"','""') '"']; end
 end
 fprintf(fid,'%s%c',strjoin(values,','),10);
end
end
function write_macro(fid,name,value)
fprintf(fid,'%s%c',[char(92) 'newcommand{' char(92) name '}{' value '}'],10);
end
