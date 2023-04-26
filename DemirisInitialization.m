function W = DemirisInitialization (params)
% function W = DemirisInitialization (params)

    neuronNumber    = params.neuronNumber;
    maxEigenValue   = params.maxEigenValue;
    W = zeros(neuronNumber,neuronNumber);
    percNonZero = params.percNonZero;
    for i=1:neuronNumber
        for j=1:neuronNumber
            W(i,j) = rand(1,1);
            W(i,j) = (W(i,j)<percNonZero);        
        end
    end

    W = randn(neuronNumber) .* W;

    % enforce desired spectral radius
    [~,d] = eig(W);
    d = diag(abs(d));
    d = max(d,[],1);
    W = W*maxEigenValue/d;
end