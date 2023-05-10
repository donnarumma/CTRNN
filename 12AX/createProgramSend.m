function [net] = createProgramSend(tau)
% function [net] = createProgramSend(tau)
    n=6;
    net=createCTRNN(n,2);
    fp = ones(1,n)*0.5;
    net.internalWMatrix=eye(n);
    v = 1.4040100352169025;
    b = -1.2020050176152283;
    net.externalWMatrix=[   v 0; ...
                            0 v; ...
                            v 0; ...
                            v 0; ...
                            0 v; ...
                            0 v];
    net.biases=ones(1,n)*b;
    % net.dt = 0.01;
    if nargin < 1
       net.tau=1;
    else 
        net.tau=tau;
    end
    net.initialOutValues= fp;
    %net.initialYValues=[-2 -1];
    net.asyntMod=0;
    net.asynTime=2000; 
    net.indexOut = 1;
    net.outs(net.indexOut,:) = net.initialOutValues;
return