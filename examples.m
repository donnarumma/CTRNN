clear all; close all;
net                 =createCTRNN(2,2); % default 2x2
net.internalWMatrix =[ 3 1;-1 3];
net.externalWMatrix =[-2 0;0 -1];
net.biases          =[ 0  0];
net.tau             =[10 10];
net.initialOutValues=[ 0  0];
net.externalInput   =[0.9 0.9];
net.asympMod        =1;

net.asymTime        =2000; 

tic
out_net = runCTRNN(net);
toc
% mulation procedure
netMul  =mulation(net   ,'internal',1,1,[-5 5],1);
netMul  =mulation(netMul,'internal',2,2,[-5 5],1);
netMul  =mulation(netMul,'external',1,1,[-5 5],1);
netMul  =mulation(netMul,'external',2,2,[-5 5],1);
netMul.asynTime=5000;

tic
out_netMul=runCTRNN(netMul);
toc 

figure;
hold on; box on; grid on;
plot(out_net(:,1), out_net(:,2),'r*') 
plot(out_netMul(:,1), out_netMul(:,2),'bo')
title('Red: no muled network, Blue: muled network');