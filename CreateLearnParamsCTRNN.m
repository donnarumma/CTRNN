function learnParams=createLearnParamsBPTT(trainSet,trainTarget,derivActFunctions,validSet,validTarget)
%function learnParams=createLearnParams(trainSet,trainTarget,derActivationFunctions,learningAlgorithm,validSet,validTarget)
% -- trainSet is a nxd matrix
% -- trainTarget is a nxc matrix
% -- derActivationFunctions is a cellarray of function pointers
% -- (optional) validSet is a mxd matrix
% -- (optional) validTarget is a mxc matrix
learnParams                     = struct();
learnParams.trainSet            = trainSet;
learnParams.trainTarget         = trainTarget;
learnParams.derivActFunctions   = derivActFunctions;
learnParams.maxEpochs           = 100;
learnParams.errorFunction       = @SQError;
learnParams.derivErrorFunction  = @derSQError;
learnParams.weightAndBiasUpdate = 1;
learnParams.stopCriteria        = @stopCriteria;
learnParams.stopIterToll        = 10;

if nargin < 5
    learnParams.useValidation=false;
else
    learnParams.useValidation = true;
    learnParams.validSet      = validSet;
    learnParams.validTarget   = validTarget;
end

