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
save([dr 'W.txt']    ,'-ascii','W'    );
save([dr 'WE.txt']   ,'-ascii','WE'   );
save([dr 'theta.txt'],'-ascii','theta');
save([dr 'tau.txt']  ,'-ascii','tau'  );
save([dr 'in.txt']   ,'-ascii','in'   );
save([dr 'x0.txt']   ,'-ascii','x0'   );
return