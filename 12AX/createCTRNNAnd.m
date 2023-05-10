% simulate And
function [net] = createCTRNNAnd(tau)
    net=createCTRNN(1,2);
    fp = 0;%0.5;
    net.internalWMatrix=5;
    net.externalWMatrix=[3 3];
    net.biases= -7;
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
return;