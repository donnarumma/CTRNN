function [inputs] = generateInputs (progs, 	...
				    howmany, 	...
				    duration,   ...
				    sigma_duration)

if nargin < 2
  howmany = 50;
end
if nargin < 3 
  duration = 200;
end
if nargin < 4
  sigma_duration = 50;
end
for k = 1:howmany
	inputs_k = [];
	for i = 1:length (progs)
		if progs(i) == 1;
			inputs_k = [inputs_k sequenceRight(duration,sigma_duration)];
		else
			inputs_k = [inputs_k sequenceLeft(duration,sigma_duration)];
		end
	end
	inputs{k}=sigmoid(inputs_k);
end
	