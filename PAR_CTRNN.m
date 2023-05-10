function   params=PAR_CTRNN
% function params=PAR_CTRNN
params.outputFun   =str2func('sigmoid');
params.dt      = 0.1;
params.tau          = 1;
params.modelType   ='CTRNN';
params.asympMod    = 0;
params.asymTime    = 100;
params.AM_STEP      = 100;
params.AM_FP_STEP   = 4;
params.NOISE        = str2func('NET_Noise');
params.NOISE_MU     = 0;
params.NOISE_SIGMA  = 0;%0.1;
params.TangentSpace      = str2func('TangentSpace');
params.Tangent2DSpaceGrid=str2func('computeTangentSpaceGrid');
params.Obs          = 1;



    




