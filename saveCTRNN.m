function [] = saveCTRNN(net,dr)
if nargin < 2
    dr = './';
end
W       = net.internalWMatrix;
WE      = net.externalWMatrix;
theta   = net.biases;
tau     = net.tau;
t       = net.asynTime;
in      = repmat(net.externalInput,t,1);
x0      = net.initialOutValues;

saveBinDoubleMatrix(W,      [dr 'W.bin']    );
saveBinDoubleMatrix(WE,     [dr 'WE.bin']   );
saveBinDoubleMatrix(theta,  [dr 'theta.bin']);
saveBinDoubleMatrix(tau,    [dr 'tau.bin']  );
saveBinDoubleMatrix(in,     [dr 'in.bin']   );
saveBinDoubleMatrix(x0,     [dr 'x0.bin']   );

% save([dr 'W.txt']    ,'-ascii','-double','W'    );
% save([dr 'WE.txt']   ,'-ascii','-double','WE'   );
% save([dr 'theta.txt'],'-ascii','-double','theta');
% save([dr 'tau.txt']  ,'-ascii','-double','tau'  );
% save([dr 'in.txt']   ,'-ascii','-double','in'   );
% save([dr 'x0.txt']   ,'-ascii','-double','x0'   );
return