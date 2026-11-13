function stats=test_cpb_integration()
%TEST_CPB_INTEGRATION Independent indexing, recognition, and trajectory checks.
root=fileparts(mfilename('fullpath'));
oldpath=path; restore=onCleanup(@()path(oldpath)); %#ok<NASGU>
addpath(root); addpath(fullfile(root,'vendor_mpb'));
c=cpb_config(); p=mpb_parameters();
saved=load(fullfile(root,'training_fixture.mat'),'fixture'); f=saved.fixture;
for r=[1,17,60]
 for b=1:6
  u=reshape(f.teacher_u(r,b,:,:),c.max_queries,3);
  for j=1:c.max_queries
   for h=1:3
    assert(u(j,h)==f.teacher_u(r,b,j,h));
    tc=f.task_code(r,f.order(r,b));
    L=cpb_query_model(h,c);
    ref=find(f.teacher_u(r,b,j,h)<=cumsum(L(:,tc)),1);
    assert(cpb_teacher_reply(j,h,u,tc,c)==ref);
   end
  end
 end
end
fprintf('Teacher tensor indexing and curriculum/truth mapping passed: 648 checked query positions.\n');
counts=ones(6)+24*eye(6); codes=eye(6); motor=struct('success',eye(6),'cost',zeros(6));
for stage=1:6
 seen=(1:6)<=stage;
 v=cpb_eval_memory(codes,counts,seen,1:6,1:6,motor,p,c);
 expected=c.cue_reliability+(6-stage)*(1-c.cue_reliability)/(5*stage);
 assert(abs(mean(v(:,1))-expected)<1e-12);
 assert(all(abs(v(:,2)-1)<1e-12));
 fprintf('  Stage %d independent exact cue-mixture success %.12f.\n',stage,mean(v(:,1)));
end
s=load(fullfile(root,'vendor_mpb','layout_manifest.mat'));
layouts=s.layouts(~strcmp({s.layouts.split},'development')); mc=mpb_protocol();
max_cost_difference=0; n=0;
for li=[1,numel(layouts)]
 for selected=1:6
  for truth=1:6
   a=mpb_rollout(layouts(li),selected,truth,'interpreted',p,mc,true);
   b=mpb_rollout(layouts(li),selected,truth,'compiled',p,mc,true);
   assert(a.success==b.success && a.wrong_goal==b.wrong_goal);
   assert(isequal(a.trace(:,1:13),b.trace(:,1:13)));
   max_cost_difference=max(max_cost_difference,abs(a.motor_cost-b.motor_cost)); n=n+1;
  end
 end
end
assert(max_cost_difference==0);
fprintf('Independent actual compiled/interpreted closed-loop trace comparison: %d pairs, exact actions/states/outcomes and zero cost difference.\n',n);

stats=struct('teacher_cases',648,'cue_mixtures',6,'trajectory_pairs',n,'max_cost_difference',max_cost_difference,'passed',true);
end
