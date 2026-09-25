function Y = t3_programmed_linear(values, coefficients, side)
%T3_PROGRAMMED_LINEAR Exact scalar linear operator with coefficient inputs.
% right: Y = values * coefficients; left: Y = coefficients * values.
% The loop structure stays fixed when the coefficient array changes.
% In attention, K' is the right-side score program and A the left-side
% value-combination program. Multiplication here is exact arithmetic, NOT
% the historical approximate CTRNN mulation construction.
assert(isnumeric(values) && isnumeric(coefficients) && ...
    all(isfinite(values(:))) && all(isfinite(coefficients(:))), ...
    't3:Finite', 'Inputs must be finite numeric matrices.');
if strcmp(side, 'right')
    assert(size(values, 2) == size(coefficients, 1), 't3:Shape', 'Right program shape mismatch.');
    Y = zeros(size(values, 1), size(coefficients, 2));
    for row = 1:size(Y, 1)
        for column = 1:size(Y, 2)
            for index = 1:size(values, 2)
                Y(row, column) = Y(row, column) + values(row, index) * coefficients(index, column);
            end
        end
    end
elseif strcmp(side, 'left')
    assert(size(coefficients, 2) == size(values, 1), 't3:Shape', 'Left program shape mismatch.');
    Y = zeros(size(coefficients, 1), size(values, 2));
    for row = 1:size(Y, 1)
        for column = 1:size(Y, 2)
            for index = 1:size(values, 1)
                Y(row, column) = Y(row, column) + coefficients(row, index) * values(index, column);
            end
        end
    end
else
    error('t3:Side', 'side must be right or left.');
end
end
