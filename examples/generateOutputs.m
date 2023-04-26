function [outputs] = generateOutputs (inputs,     ...
                                      progs,      ...
                                      threshUp,	  ...
                                      threshDown)
if nargin < 4
    threshUp	=  0.8;
end
if nargin < 5
    threshDown	=  0.2;
end
right = 1;
left  = 0;
howmany = length(inputs);
for k = 1:howmany
    inputs_k = inputs{k};
	outputs_k = zeros(1,howmany);
    i_current_value = 1;
	for i = 1:length (progs)
        
        current_prog    = progs(i);
        overTreshUp     = false;
        underTreshUp    = false;
        overTreshDown   = false;
        underTreshDown  = false;
        
        if current_prog == right;
            while ~ underTreshUp
                current_value = inputs_k(i_current_value);
                
                outputs_k (i_current_value) = right;
                if current_value > threshUp
                    overTreshUp = true;
                end
                if overTreshUp
                    if current_value < threshUp
%                         current_value
                        underTreshUp = true;
                    end
                end
                i_current_value = i_current_value + 1;
            end
        elseif current_prog == left;
            while ~ overTreshDown
                current_value = inputs_k(i_current_value);
                outputs_k (i_current_value) = left;
                if current_value < threshDown
                    underTreshDown = true;
                end
                if underTreshDown
                    if current_value > threshDown
                        overTreshDown = true;
%                         current_value
                    end
                end
                i_current_value = i_current_value + 1;
            end
        end
    end
    outputs{k} = outputs_k;
end