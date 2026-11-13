function [terminal,cost,steps] = mpb_predict_costs(layout,p,c)
%MPB_PREDICT_COSTS Independent analytic model of exact GO + ordered TOUCH.
% Rows are selected programs; columns are hidden true routes.
terminal=ones(6); cost=zeros(6); steps=zeros(6);
for selected=1:6
    for truth=1:6
        pos=layout.start; moves=0; touches=0;
        for stage=1:4
            goal=p.routes(selected,stage);
            moves=moves+sum(abs(pos-layout.landmarks(goal,:)));
            touches=touches+1; pos=layout.landmarks(goal,:);
            if goal~=p.routes(truth,stage)
                assert(stage<=2,'mpb:Category','A permutation of three goals cannot first disagree at stage 3.');
                terminal(selected,truth)=stage+1;
                break
            end
        end
        cost(selected,truth)=c.move_cost*moves+c.touch_cost*touches;
        steps(selected,truth)=moves+touches;
    end
end
end
