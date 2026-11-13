function out=cpb_motor_validation(p,quick)
%CPB_MOTOR_VALIDATION Closed-loop execution and analytic dictionary comparison.
root=fileparts(mfilename('fullpath'));
s=load(fullfile(root,'vendor_mpb','layout_manifest.mat'),'layouts');
layouts=s.layouts(~strcmp({s.layouts.split},'development'));
if quick, layouts=layouts(1:2); end
mc=mpb_protocol(); costs=zeros(6); success=zeros(6); maxerr=0; maxcost=0; n=0;
for k=1:numel(layouts)
 [terminal,cost]=mpb_predict_costs(layouts(k),p,mc);
 for selected=1:6
  for truth=1:6
   run=mpb_rollout(layouts(k),selected,truth,'interpreted',p,mc,false);
   assert(run.success==(terminal(selected,truth)==1),'cpb:Motor','Wrong route outcome.');
   assert(~run.timeout,'cpb:Motor','Unexpected timeout.');
   maxcost=max(maxcost,abs(run.motor_cost-cost(selected,truth)));
   maxerr=max(maxerr,run.max_compiled_error); n=n+1;
  end
 end
 costs=costs+cost; success=success+(terminal==1);
end
out=struct('n_layouts',numel(layouts),'n_rollouts',n, ...
 'max_compiled_command_error',maxerr,'max_dictionary_cost_error',maxcost, ...
 'cost',costs/numel(layouts),'success',success/numel(layouts), ...
 'layouts',{layouts},'dictionary_definition', ...
 'Direct GO(goal) calls in the selected order, with Manhattan+TOUCH costs and first wrong TOUCH termination.');
assert(isequal(out.success,eye(6)) && maxerr<1e-12 && maxcost<1e-12);
end
