function [net] = createAX()
% function [net] = createAX(tau)
% alphabet = 1 A X 2 B Y C D 
    net=createCTRNN(2,8);
    
    fp = [0 0];%0.5;
    net.internalWMatrix=[ 5 0; ...
                          6 5];
                        %1 A X
    net.externalWMatrix=[3 3 0 0 0 0 0 0; ...
                         0 0 3 0 0 0 0 0];
    net.biases  = [-7 -7];
    % net.dt = 0.01;
    
    net.tau= [9 3];
     
    net.initialOutValues= fp;
    %net.initialYValues=[-2 -1];
    net.asyntMod=0;
    net.asynTime=2000; 
    net.indexOut = 1;
    net.outs(net.indexOut,:) = net.initialOutValues;
return