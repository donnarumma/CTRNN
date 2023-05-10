function [out,limits,net]=demo_ADM()
t1              =  1;
t2              =  6;
w11             =  8;
w12             = -6;
w21             = 16;
w22             = -2;
b1              = -0.34;
b2              = -2.5;
N_Cells         = 2;
N_Inputs        = 0;
DELTA_T         = 1/100;  % forward Euler parameter 0.01 s
MAX_TIME        = 1000000;
NOISE_SIGMA     = 0.1;
x10             = 0.5;
x20             = 0.1;

net             = createCTRNN(N_Cells,N_Inputs);
net.Obs         = 1:2;
% net.externalWMatrix =[];
net.internalWMatrix =[w11  w12;  ...
                      w21   w22];
net.biases          =[ b1  b2 ];
net.tau             =[ t1  t2 ];
net.initialOutValues=[x10 x20 ];
net.dt              = DELTA_T;

net.NOISE_SIGMA     = NOISE_SIGMA;
t=tic;
net.externalInput   =zeros(MAX_TIME,1);
out                 =runCTRNN(net,MAX_TIME);
fprintf('Standart run net: Elapsed Time %.3f s\n',toc(t));

net_lim                 =net;
net_lim.NOISE_SIGMA     =0;
net_lim.initialOutValues=[0.1,0.5];
t=tic;
LIM_TIME                =100000;
net_lim.externalInput   =zeros(LIM_TIME,1);
limits{1}             =runCTRNN(net_lim,LIM_TIME);
fprintf('Standart run net_lim 1: Elapsed Time %.3f s\n',toc(t));

net_lim.initialOutValues=[0.5,0.1];
limits{2}             =runCTRNN(net_lim,LIM_TIME);
fprintf('Standart run net_lim 2: Elapsed Time %.3f s\n',toc(t));

limits{1}=limits{1}(end-LIM_TIME/10:end,:);
limits{2}=limits{2}(end-LIM_TIME/10:end,:);
% limits{1}=limits{1}';
% limits{2}=limits{2}';
