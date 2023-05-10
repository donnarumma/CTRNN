function [net] = createNeuronCompiled(tau)
% function [net] = createCTRNNNot(tau)
    net=createCTRNN(1,1);
    fp = 0.5;
    net.internalWMatrix=1;
    net.externalWMatrix=1.4040100352169025;
    net.biases=-1.2020050176152283;
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