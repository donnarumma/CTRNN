function observed=cpb_teacher_reply(query_index,phase,uniforms,true_code,c)
%CPB_TEACHER_REPLY Environment-only data generator. Hidden code stays here.
L=cpb_query_model(phase,c);
observed=find(uniforms(query_index,phase)<=cumsum(L(:,true_code)),1);
end
