function fixture=cpb_make_fixture(c)
%CPB_MAKE_FIXTURE Explicit fixture creation; normal runs load the saved bytes.
old=rng(); restore=onCleanup(@()rng(old)); %#ok<NASGU>
rng(c.seed,'twister');
fixture.version=c.version; fixture.seed=c.seed;
fixture.task_code=zeros(c.n_replications,6);
fixture.task_cue=zeros(c.n_replications,6);
fixture.order=zeros(c.n_replications,6);
fixture.cues=zeros(c.n_replications,6,c.cues_per_block);
fixture.teacher_u=rand(c.n_replications,6,c.max_queries,3);
fixture.random_phases=randi(3,c.n_replications,6,c.max_queries);
for group=1:ceil(c.n_replications/6)
 base=randperm(6);
 for shift=0:5
  r=(group-1)*6+shift+1; if r>c.n_replications, break; end
  fixture.order(r,:)=circshift(base,[0,shift]);
 end
end
for r=1:c.n_replications
 fixture.task_code(r,:)=randperm(6); fixture.task_cue(r,:)=randperm(6);
 for b=1:6
  task=fixture.order(r,b); likelihood=ones(1,6)*(1-c.cue_reliability)/5;
  likelihood(fixture.task_cue(r,task))=c.cue_reliability;
  for j=1:c.cues_per_block
   fixture.cues(r,b,j)=find(rand()<=cumsum(likelihood),1);
  end
 end
end
fixture.config=c;
end
