%% Questo script addestra una CTRNN ad avere come punti di equilibrio
% i coefficienti delle azioni prototipo 


clear all;

addpath('/home/guglielmo/Dropbox/matlab_code/CTRNN/');
addpath('/home/guglielmo/Dropbox/matlab_code/Common/');
addpath('/home/guglielmo/Dropbox/matlab_code/Mirror+Mixture/MixtureNetwork/');
addpath('/home/guglielmo/Dropbox/matlab_code/Mirror+Mixture/MirrorArchitecture/');

%% Creating the input for the network

Testname = 'Test12';

data_motor = load(['/home/guglielmo/Desktop/lavoro/MotorData/' Testname '/data_motor.mat']);

choosen_tree = [1,4,1];

num_princ_comp = 2;
num_split       = data_motor.Alltrees(choosen_tree(1),:);
num_level       = size(num_split,2)+1;
num_synergies   = [1,2,6];

toll = 1.5;

num_action_per_class = data_motor.numElem4Class;

treeData = generateTree(num_split, ones(1,num_level));

indForLevel =findIndForLevel(treeData.depth);

act_coefficient = sorting_act_coefficient(data_motor,choosen_tree,indForLevel);

act_coefficient = normalizing_motor(act_coefficient,num_synergies);

%% Costruisco il training ed il validation
[prototipPG, prototipWH, prototip_usagePG, prototip_usageWH] = Calculating_Proto_action(act_coefficient, num_action_per_class, toll);


trainSet(:,1) = prototipPG';
trainSet(:,2) = prototipWH';

trainTarget = trainSet;

validationElement = 10;
validSet = repmat(trainSet,1,validationElement);
validTarget = validSet;

sigma = 0.1;

validSet = validSet + sigma.*validSet.*randn(size(validSet));


%% Creating the CTRNN

numUnits = sum(num_synergies);
numExternalInput = sum(num_synergies);
biases = zeros(1,numUnits);

net=createCTRNN(numUnits,numExternalInput,biases);

net.timeStep = 100; % 10 volte tau

%% Learn Params

derivActFunctions = @derivSigmoid;
learnParams=CreateLearnParamsBPTT(trainSet,trainTarget,derivActFunctions,validSet,validTarget);
learnParams=setLearnParams(learnParams,'eta',0.01);

%% Learning

net = backpropagationTT5(net,learnParams);

