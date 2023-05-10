% function demo_2neurons_plus_mul()
% clear all; close all;
net                 =createCTRNN(2,2);
net.internalWMatrix =[ 3   1   ;  -1   3];
net.externalWMatrix =[-2   0   ;   0  -1];
net.biases          =[ 0   0  ];
net.tau             =[10  10  ];
net.initialOutValues=[ 0   0  ];
net.externalInput   =[ 0.9 0.9];
net.NOISE_SIGMA     =0.1;
N_TIMES = 2000;
% N_TIMES = 2000;

NET     =net;
netMul  =net;

t=tic;
NET.externalInput   =repmat(net.externalInput,N_TIMES,1);
out_NET             =runCTRNN(NET,N_TIMES);
fprintf('Standart run net: Elapsed Time %.3f s\n',toc(t));

t=tic;
out_net=nan(N_TIMES,net.numUnits);
for iT=1:N_TIMES
    [out_net(iT,:)]=runCTRNN(net,1);
    net.initialOutValues=out_net(iT,:);
end
fprintf('Iterated run net: Elapsed Time %.3f s\n',toc(t));

netMul  =mulation(netMul,'internal',1,1,[-5 5],1);
netMul  =mulation(netMul,'internal',2,2,[-5 5],1);
netMul  =mulation(netMul,'external',1,1,[-5 5],1);
netMul  =mulation(netMul,'external',2,2,[-5 5],1);

% N_TIMES = N_TIMES*2;%5000;
NETMUL  =netMul;

t=tic;
NETMUL.externalInput    =repmat(netMul.externalInput,N_TIMES,1);
OUT_NETMUL              =runCTRNN(NETMUL,N_TIMES);
fprintf('Standard run mul net: Elapsed Time %.3f s\n',toc(t));

t=tic;
out_netMul=nan(N_TIMES,netMul.numUnits);
for iT=1:N_TIMES
    [out_netMul(iT,:)]=runCTRNN(netMul,1);
    netMul.initialOutValues=out_netMul(iT,:);
end
fprintf('Iterated run mul net: Elapsed Time %.3f s\n',toc(t));


figure;
isequal(out_netMul,OUT_NETMUL)
isequal(out_NET,out_net)
hold on; box on; grid on;
plot(   out_net(:,1),    out_net(:,2),'r*') 
plot(out_netMul(:,1), out_netMul(:,2),'bo')
title('Red: no muled network, Blue: muled network');