function [net] = createMemory12(tau)
% function [net] = createMemory12(tau)
    net=createCTRNN(2,8);
    fp = [0.5, 0.5];
    net.internalWMatrix=[20  0; ...
                          0 20];
                      
                         %1 A X  2 B Y C D
    net.externalWMatrix=[+7 0 0 -7 0 0 0 0; ...
                         -7 0 0 +7 0 0 0 0];
    net.biases= [-10 -10];
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