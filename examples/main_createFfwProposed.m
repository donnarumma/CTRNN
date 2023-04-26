addpath /media/Data/Boxes/BoxGmail/Dropbox/NetworkCode/FfwProposed/;
addpath /media/Data/Boxes/BoxInwind/Dropbox/matlab_code/CTRNN/;
addpath /media/Data/Boxes/BoxInwind/Dropbox/matlab_code/Opt/;
addpath /media/Data/Boxes/BoxInwind/Dropbox/matlab_code/Common/;

numOfNeuronForLayer = [2 , 3,  2];
net     = createFfwProposed(numOfNeuronForLayer);

net     = ffwForwardStepProposed(net,[0,0;1,1;2,5]);

% net     = ffwForwardStepProposed(net,[1,1]);
learnParams = createLearnParams();