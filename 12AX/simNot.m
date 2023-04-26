% simulate Not
net=createCTRNN(1,1);
fp = 0.5;
net.internalWMatrix=7;
net.externalWMatrix=-14;
net.biases= 0;
% net.dt = 0.01;
net.tau=1;
net.initialOutValues= fp;
%net.initialYValues=[-2 -1];
net.asyntMod=1;
net.asynTime=2000; 

figure; hold on; grid on;

net.externalInput=0;
[out]=runCTRNN(net);

plot(out,'r*')
% return
net.externalInput=1;
[out]=runCTRNN(net);

plot(out,'g*')

legend ('input 0','input 1');
hold off;
return;
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