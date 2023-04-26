function [net] = createBY()
% function [net] = createBY(tau)
% alphabet = 1 A X 2 B Y C D 
    net=createCTRNN(2,10);
    
    fp = [0 0];%0.5;
    net.internalWMatrix=[ 5 0; ...
                          4 5];
                        %m1 m2 1 A X 2 B Y C D 
    net.externalWMatrix=[ 0  3 0 0 0 0 3 0 0 0; ...
                          0  0 0 0 0 0 0 3 0 0];
    net.biases  = [-7 -7];
    % net.dt = 0.01;
    
    tau = [9,3];
%     tau = [90,30];
    net.tau = tau;
     
    net.initialOutValues= fp;
    %net.initialYValues=[-2 -1];
    net.asyntMod=0;
    net.asynTime=2000; 
    net.indexOut = 1;
    net.outs(net.indexOut,:) = net.initialOutValues;
return