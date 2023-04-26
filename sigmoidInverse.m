%% function y=sigmoidInverse(x)
% inverse of sigmoid
function y=sigmoidInverse(x)
    y = log(x./(1-x));
end

% y=1 ./ (1+exp(-x));
%(1+exp(-x)*y=1);
% (y+exp(-x)*y=1);