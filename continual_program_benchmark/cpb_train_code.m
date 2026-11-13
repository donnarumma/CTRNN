function [q,trace]=cpb_train_code(mode,teacher,random_phases,n_yoked,c,p)
%CPB_TRAIN_CODE Learner only sees replies to questions it actually requests.
% teacher is an environment callback: observed=teacher(query_index,phase).
q=ones(1,6)/6; trace=struct('step',{},'phase',{},'observation',{}, ...
    'posterior',{},'entropy',{},'G',{},'free_energy_error',{});
if strcmp(mode,'random_yoked'), budget=n_yoked; else, budget=c.max_queries; end
for j=1:budget
 choice=cpb_select_query(q,c);
 if strcmp(mode,'active_archive')
  h=choice.phase; if h==0, break; end
 elseif strcmp(mode,'active_fixed')
  gs=choice.G(2:4); h=find(gs<=min(gs)+1e-12,1);
 else
  h=random_phases(j);
 end
 observed=teacher(j,h);
 update=cpb_code_update(q,h,observed,c,p); q=update.posterior;
 trace(end+1)=struct('step',j,'phase',h,'observation',observed, ...
    'posterior',q,'entropy',-sum(q(q>0).*log(q(q>0))), ...
    'G',choice.G(h+1),'free_energy_error',update.free_energy_identity_error); %#ok<AGROW>
end
end
