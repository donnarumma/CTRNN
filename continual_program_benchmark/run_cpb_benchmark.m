function report=run_cpb_benchmark(cfg)
%RUN_CPB_BENCHMARK Continual acquisition with distinct code and retrieval memory.
if nargin<1, cfg=struct(); end
if ~isfield(cfg,'quick'), cfg.quick=false; end
if ~isfield(cfg,'make_plots'), cfg.make_plots=true; end
if ~isfield(cfg,'save_results'), cfg.save_results=true; end
if ~isfield(cfg,'use_spm_reference'), cfg.use_spm_reference=true; end
root=fileparts(mfilename('fullpath'));
if ~isfield(cfg,'output_dir'), cfg.output_dir=fullfile(root,'results'); end
oldpath=path; restore=onCleanup(@()path(oldpath)); %#ok<NASGU>
addpath(root); addpath(fullfile(root,'vendor_mpb'));
c=cpb_config(); p=mpb_parameters(); initial_p=p;
f=load(fullfile(root,'training_fixture.mat'),'fixture'); f=f.fixture;
assert(strcmp(f.version,c.version) && isequal(f.config.routes,c.routes));
for field={'cue_reliability','demo_reliability','query_cost','max_queries','cues_per_block','dirichlet_alpha','seed'}
 assert(isequal(f.config.(field{1}),c.(field{1})),'cpb:Fixture','Protocol changed; create a new fixture version.');
end
if cfg.quick, c.n_replications=6; end
report.config=c; report.conditions=c.conditions;
report.metadata=struct('version',c.version,'quick',cfg.quick,'runtime',version, ...
 'training_boundaries_known',true,'test_task_ids_provided',false, ...
 'teacher_model_known',true,'cue_likelihood_learned',true,'fixed_interpreter',true, ...
 'growing_program_memory',true,'multi_agent',false,'training_fixture','training_fixture.mat');
assert(size(f.order,1)>=c.n_replications && size(f.cues,3)==c.cues_per_block && size(f.teacher_u,3)==c.max_queries);
report.core_tests=test_cpb_core();
report.integration_tests=test_cpb_integration();
fprintf('Core tests passed. Checking closed-loop motor execution...\n');
report.motor=cpb_motor_validation(p,cfg.quick);
rows=struct('replication',{},'stage',{},'task_id',{},'learned_at',{},'condition',{}, ...
 'success',{},'oracle_context_success',{},'recognition_accuracy',{},'nll',{}, ...
 'brier',{},'motor_cost',{},'code_nll',{},'oracle_program_success',{});
queries=struct('replication',{},'stage',{},'condition',{},'nqueries',{},'cost',{}, ...
 'code_success',{},'code_nll',{},'entropy',{});
acquisition=struct('replication',{},'stage',{},'condition',{},'step',{},'phase',{}, ...
 'observation',{},'code_success',{},'code_nll',{},'brier',{},'entropy',{});
checkpoints=struct('replication',{},'stage',{},'order',{},'true_codes',{},'cue_peaks',{}, ...
 'cue_counts',{},'q_active',{},'q_random_yoked',{},'q_active_fixed',{},'q_random_fixed',{});
max_code_drift=0; max_cue_drift=0; max_F_error=0;
for r=1:c.n_replications
 bank=cell(1,4); for z=1:4, bank{z}=ones(6)/6; end
 counts=ones(6)*c.dirichlet_alpha;
 order=f.order(r,:); true_codes=f.task_code(r,order); peaks=f.task_cue(r,order);
 for b=1:6
  previous=bank; previous_counts=counts;
  observed=reshape(f.cues(r,b,:),1,[]);
  for symbol=observed, counts(b,symbol)=counts(b,symbol)+1; end
  uniforms=reshape(f.teacher_u(r,b,:,:),c.max_queries,3);
  random_phases=reshape(f.random_phases(r,b,:),1,[]);
  teacher=@(j,h)cpb_teacher_reply(j,h,uniforms,true_codes(b),c);
  n_yoked=0;
  for z=1:4
   mode=c.conditions{z};
   [posterior,tr]=cpb_train_code(mode,teacher,random_phases,n_yoked,c,p);
   bank{z}(b,:)=posterior;
   if z==1, n_yoked=numel(tr); end
   [~,code]=max(posterior);
   queries(end+1)=struct('replication',r,'stage',b,'condition',mode, ...
    'nqueries',numel(tr),'cost',numel(tr)*c.query_cost, ...
    'code_success',double(code==true_codes(b)),'code_nll',-log(posterior(true_codes(b))), ...
    'entropy',-sum(posterior.*log(posterior))); %#ok<AGROW>
   for step=0:numel(tr)
    if step==0, q=ones(1,6)/6; h=0; obs=0;
    else, q=tr(step).posterior; h=tr(step).phase; obs=tr(step).observation;
     max_F_error=max(max_F_error,tr(step).free_energy_error);
    end
    [~,code]=max(q); target=zeros(1,6); target(true_codes(b))=1;
    acquisition(end+1)=struct('replication',r,'stage',b,'condition',mode,'step',step, ...
      'phase',h,'observation',obs,'code_success',double(code==true_codes(b)), ...
      'code_nll',-log(q(true_codes(b))),'brier',sum((q-target).^2), ...
      'entropy',-sum(q(q>0).*log(q(q>0)))); %#ok<AGROW>
   end
  end
  if b>1
   max_cue_drift=max(max_cue_drift,max(max(abs(counts(1:b-1,:)-previous_counts(1:b-1,:)))));
   for z=1:4
    max_code_drift=max(max_code_drift,max(max(abs(bank{z}(1:b-1,:)-previous{z}(1:b-1,:)))));
   end
  end
  seen=(1:6)<=b;
  for z=1:numel(c.conditions)
   cue=counts;
   if z<=4, codes=bank{z}; else, codes=bank{1}; end
   if strcmp(c.conditions{z},'reset_old_codes'), codes(1:b-1,:)=ones(b-1,6)/6; end
   if strcmp(c.conditions{z},'reset_old_router'), cue(1:b-1,:)=c.dirichlet_alpha; end
   values=cpb_eval_memory(codes,cue,seen,true_codes,peaks,report.motor,p,c);
   for j=1:b
    v=values(j,:);
    rows(end+1)=struct('replication',r,'stage',b,'task_id',order(j),'learned_at',j, ...
     'condition',c.conditions{z},'success',v(1),'oracle_context_success',v(2), ...
     'recognition_accuracy',v(3),'nll',v(4),'brier',v(5),'motor_cost',v(6), ...
     'code_nll',v(7),'oracle_program_success',v(8)); %#ok<AGROW>
   end
  end
  checkpoints(end+1)=struct('replication',r,'stage',b,'order',order,'true_codes',true_codes, ...
   'cue_peaks',peaks,'cue_counts',counts,'q_active',bank{1},'q_random_yoked',bank{2}, ...
   'q_active_fixed',bank{3},'q_random_fixed',bank{4}); %#ok<AGROW>
 end
 if mod(r,6)==0, fprintf('Completed %d/%d frozen learning streams.\n',r,c.n_replications); end
end
report.rows=rows; report.query_rows=queries; report.acquisition=acquisition;
report.checkpoints=checkpoints; report.fixture=f;
report.summary=struct('condition',{},'success',{},'oracle_context_success',{}, ...
 'recognition_accuracy',{},'nll',{},'brier',{},'motor_cost',{},'code_nll',{});
for z=1:numel(c.conditions)
 match=strcmp({rows.condition},c.conditions{z}) & [rows.stage]==6;
 a=rows(match);
 report.summary(end+1)=struct('condition',c.conditions{z},'success',mean([a.success]), ...
 'oracle_context_success',mean([a.oracle_context_success]),'recognition_accuracy',mean([a.recognition_accuracy]), ...
 'nll',mean([a.nll]),'brier',mean([a.brier]),'motor_cost',mean([a.motor_cost]),'code_nll',mean([a.code_nll])); %#ok<AGROW>
end
assert(isequal(p,initial_p),'cpb:Weights','Interpreter parameters changed.');
assert(max_code_drift==0 && max_cue_drift==0,'cpb:Archive','Stored memory changed.');
% Matched executors have the same selected programs, certified by motor validation.
for z=5:6
 assert(abs(report.summary(z).success-report.summary(1).success)<1e-14);
end
for b=1:6
 for r=1:c.n_replications
  a=queries([queries.replication]==r & [queries.stage]==b);
  assert(a(1).nqueries==a(2).nqueries && a(3).nqueries==c.max_queries && a(4).nqueries==c.max_queries);
 end
end
assert(all([rows.success]>=-1e-12 & [rows.success]<=1+1e-12));
report.validation=struct('passed',true,'max_stored_code_drift',max_code_drift, ...
 'max_stored_cue_count_drift',max_cue_drift,'max_F_identity_error',max_F_error, ...
 'fixed_projection_weights',isequal(p,initial_p),'oracle_program_success',min([rows.oracle_program_success]));
if cfg.use_spm_reference, report.spm=cpb_spm_reference(report,cfg); end
if cfg.save_results
 cpb_write_results(report,cfg.output_dir);
 if cfg.make_plots
  cpb_plot_results(report,cfg.output_dir);
  cpb_plot_acquisition(report,cfg.output_dir);
 end
end
end
