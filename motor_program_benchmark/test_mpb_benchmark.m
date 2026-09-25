function checks=test_mpb_benchmark()
%TEST_MPB_BENCHMARK Integration: environment/model agreement and memory causality.
p=mpb_parameters(); c=mpb_protocol(); c.n_development=1; c.n_evaluation=1; c.n_transfer=1;
layouts=mpb_layouts(c); max_error=0; cases=0;
for li=1:numel(layouts)
    [terminal,cost,steps]=mpb_predict_costs(layouts(li),p,c);
    for chosen=1:6
        for truth=1:6
            r=mpb_rollout(layouts(li),chosen,truth,'interpreted',p,c,true);
            assert(r.success==double(terminal(chosen,truth)==1));
            assert(r.motor_steps==steps(chosen,truth));
            assert(abs(r.motor_cost-cost(chosen,truth))<1e-12);
            assert(r.timeout==0);
            max_error=max(max_error,r.max_compiled_error); cases=cases+1;
            if chosen==truth
                touches=r.trace(r.trace(:,7)==5,8)';
                assert(isequal(touches,p.routes(chosen,:)),'mpb:Order','Executor completed the wrong ordered goals.');
            end
        end
    end
end
% Same chosen program and visible state produce the same actions before feedback,
% irrespective of hidden context. The environment alone decides acceptance.
a=mpb_rollout(layouts(1),6,6,'interpreted',p,c,true);
b=mpb_rollout(layouts(1),6,1,'interpreted',p,c,true);
touch=find(b.trace(:,7)==5,1);
assert(isequal(a.trace(1:touch,7),b.trace(1:touch,7)),'mpb:Leak','Action sequence used the hidden rule before feedback.');
reset=mpb_rollout(layouts(1),6,6,'reset_memory',p,c,true);
assert(reset.success==0 && reset.wrong_goal==1,'mpb:Memory','Resetting the sequence register did not disrupt execution.');
checks=struct('cases',cases,'max_compiled_error',max_error,'hidden_rule_prefix_check',true,'passed',true);
disp(checks);
end
