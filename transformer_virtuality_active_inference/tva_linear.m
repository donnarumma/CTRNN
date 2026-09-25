function Y = tva_linear(values, coefficients, side)
% Fixed scalar multiply-accumulate operator; coefficients are runtime inputs.
% Standard floating-point arithmetic, not an approximate neural multiplier.
assert(isnumeric(values) && isreal(values) && ismatrix(values) && ...
    isnumeric(coefficients) && isreal(coefficients) && ismatrix(coefficients) && ...
    all(isfinite(values(:))) && all(isfinite(coefficients(:))), ...
    'tva:LinearInput','Values and coefficients must be finite real matrices.');
if strcmp(side,'right'), left=values; right=coefficients;
elseif strcmp(side,'left'), left=coefficients; right=values;
else, error('tva:Side','Side must be right or left.'); end
assert(size(left,2)==size(right,1),'tva:LinearShape','Incompatible matrix dimensions.');
Y=zeros(size(left,1),size(right,2));
for i=1:size(Y,1)
    for j=1:size(Y,2)
        for k=1:size(left,2), Y(i,j)=Y(i,j)+left(i,k)*right(k,j); end
    end
end
end
