function p = tva_parameters()
% Fixed, hand-chosen matrices. None are learned or changed during an episode.
p.Wq = [1/3 0 0; 1/3 0 0; 1/3 0 0];
p.Wk = sqrt(3)*[1 0 0; 1 0 0; 0 0 0];
p.Wv = eye(3); p.Wout = eye(3);
p.W1 = [eye(3) -eye(3)];
p.W2 = .2*[eye(3); -.5*eye(3)];
p.b1 = zeros(1,6); p.b2 = zeros(1,3);
p.gamma1 = ones(1,3); p.beta1 = zeros(1,3);
p.gamma2 = ones(1,3); p.beta2 = zeros(1,3);
p.Wclass = 2*eye(3); p.bclass = zeros(1,3);
p.layernorm_epsilon = 1e-5;
end
