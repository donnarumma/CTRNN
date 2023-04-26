clear all; close all;
net=createCTRNN(2,2);
net.internalWMatrix=[3 1;-1 3];
net.externalWMatrix=[-2 0;0 -1];
net.biases=[0 0];
net.tau=[10 10];
net.initialOutValues=[0 0];
%net.initialYValues=[-2 -1];
net.asymMod=1;
net.asymTime=2000; 
net.externalInput=[0.9 0.9];
[out]=runCTRNN(net);
plot(out(:,1), out(:,2),'r*') 
netMul=mulazione(net,'internal',1,1,[-5 5],1);
netMul2=mulazione(netMul,'internal',2,2,[-5 5],1);
netMul3=mulazione(netMul2,'external',1,1,[-5 5],1);
netMul4=mulazione(netMul3,'external',2,2,[-5 5],1);
netMul4.asynTime=5000;
[outNetMul4]=runCTRNN(netMul4);
hold on;
plot(outNetMul4(:,1), outNetMul4(:,2),'bo')
hold on
title('Red: no muled network Blue: muled network');