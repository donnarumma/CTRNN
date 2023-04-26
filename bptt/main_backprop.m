% mainbackprpagation
%  clear all
%  if size(tau,2) ~= numUnits
%  	tau = tau(1)*ones(1,numUnits);
%  end
dropdir = '../../';

addpath ([dropdir 'Opt/']);
addpath ([dropdir 'Common/']);
addpath ([dropdir 'CTRNN/']);
addpath ([dropdir 'datasets/']);
addpath ('/media/Data/Boxes/BoxGmail/Dropbox/NetworkCode/FfwProposed/');
[trainSet trainTarget] 	= bishopDataset(0);

numUnits		= 3;
numExternalInput	= size(trainSet,2);
biases			= zeros(1,numUnits);
internalWMatrix		= rand(numUnits,numUnits);
externalWMatrix		= rand(numUnits,numExternalInput);
%  internalWMatrix		= load('W');
%  externalWMatrix		= load('WE');

%  internalWMatrix
net			= createCTRNN(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix);
%  net
net.dt 			= 0.2;
%  net.tau		= 1;

% params              = createLearnParams(trainSet,trainTarget,{@derivSigmoid @derivIdentity});
params              = createLearnParams(trainSet,trainTarget,{@derivSigmoid @derivSigmoid});
params              = setLearnParams(params,'maxEpochs',5000);
params              = setLearnParams(params,'learningAlgorithm',@resilientBackPropagation);
params              = setLearnParams(params,'eta',0.00004);
[W,WE,errore,n]     = backpropagationTTnew ( net , params );
%  function [W,WE,errore,n]=backpropagationTT5(eta, cicli, ampiezza_intervallo,tau,numero_neuroni,matrice_target,matrice_input)
errore
net.outs(1,:)       = net.initialOutValues;
net.Ys(1,:)         = net.initialYValues;
net.externalInput	= trainSet;
tic
[out]               = runCTRNN(net);
toc
steps 			= size(trainSet);
net.indexOut		= 1; 
tic
for i=1:steps
	net.externalInput	= trainSet(i,:);
	[net]			= ctrnn_dual_fe_oneStep(net);
end
toc

net.indexOut		= 1;
net.internalWMatrix	= W;
net.externalWMatrix	= WE;
tic
for i=1:steps
	net.externalInput	= trainSet(i,:);
	[net]			= ctrnn_fe_oneStep(net);
end
toc

outN = 3;
figure;
plot(out(:,outN));

figure;
plot(net.outs(2:end,outN));

figure;
plot(sigmoid(net.Ys(2:end,outN)));
hold on;
plot(trainTarget);
