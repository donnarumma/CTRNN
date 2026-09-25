function out = mpb_rollout(layout,program_id,truth_id,backend,p,c,keep_trace)
%MPB_ROLLOUT Environment knows truth; executor only gets chosen code and state.
if nargin<7, keep_trace=false; end
pos=layout.start; accepted=0; moves=0; touches=0; success=false; wrong=false;
trace=zeros(c.max_steps,14); max_compiled_error=0; max_ideal_error=0;
program=p.routes(program_id,:);
for t=1:c.max_steps
    stage=accepted+1; before=pos;
    if strcmp(backend,'compiled')
        motor=mpb_compiled_execute(program,stage,pos,layout.landmarks,p);
    else
        intervention='none';
        if strcmp(backend,'frozen_keys'), intervention='frozen_keys'; end
        if strcmp(backend,'frozen_attention'), intervention='frozen_attention'; end
        if strcmp(backend,'reset_memory'), intervention='reset_memory'; end
        motor=mpb_execute(program,stage,pos,layout.landmarks,intervention,p);
    end
    action=motor.action; touched=0;
    if strcmp(backend,'interpreted')
        ref=mpb_compiled_execute(program,stage,pos,layout.landmarks,p);
        max_compiled_error=max(max_compiled_error,max(abs(motor.probabilities-ref.probabilities)));
        if isfield(motor,'ideal_probabilities')
            max_ideal_error=max(max_ideal_error,max(abs(motor.probabilities-motor.ideal_probabilities)));
        end
    end
    switch action
        case 1, pos(1)=max(1,pos(1)-1); moves=moves+1;
        case 2, pos(1)=min(layout.grid_size,pos(1)+1); moves=moves+1;
        case 3, pos(2)=min(layout.grid_size,pos(2)+1); moves=moves+1;
        case 4, pos(2)=max(1,pos(2)-1); moves=moves+1;
        case 5
            touches=touches+1;
            for g=1:4
                if isequal(pos,layout.landmarks(g,:)), touched=g; break; end
            end
            % This is the ONLY hidden-rule-dependent part: environment feedback.
            if touched==p.routes(truth_id,stage)
                accepted=accepted+1;
                if accepted==4, success=true; end
            else
                wrong=true;
            end
        case 6 % WAIT has a time cost, but is not selected by the ordinary GO controller.
        otherwise, error('mpb:Action','LOOK belongs to the initial policy phase.');
    end
    trace(t,:)=[t,before,pos,stage,action,touched,accepted,success,wrong,program_id,truth_id,max(motor.probabilities)];
    if success || wrong, break; end
end
out=struct('success',double(success),'wrong_goal',double(wrong),'moves',moves, ...
    'touches',touches,'motor_steps',t,'motor_cost',c.move_cost*(t-touches)+c.touch_cost*touches, ...
    'timeout',double(~success && ~wrong),'max_compiled_error',max_compiled_error, ...
    'max_ideal_error',max_ideal_error,'final_position',pos,'accepted',accepted);
if keep_trace, out.trace=trace(1:t,:); else, out.trace=[]; end
end
