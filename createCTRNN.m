function  net=createCTRNN(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix)
%function net=createCTRNN(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix)
%INPUT:         numUnits:           number of neurons (n)
%               numExternalInput:   number of external input (d)
%               biases:             (Optional) 1xn array of biases
%               internalWMatrix:    (Optional) nxn matrix of internal weights
%               externalWMatrix:    (Optional) nxd matrix of external weights
%
%OUTPUT:        net: It is a structure. The structure fields are:
%     CONNECTIONS
%               numUnits:           Scalar 
%               numExternalInput:   Scalar
%               biases:             1xd array
%               internalWMatrix:    nxn matrix
%               externalWMatrix:    nxd matrix
%               tau:                1xn array.  Default: ones(1,n);
%               dt:                 Scalar.     Default: 0.1
%               externalInput:      Nxd matrix. Default: empty array.
%     INITIAL VALUES
%               initialYValues:     1xn array.  Default: zeros(1,n).
%               initialOutValues:   1xn array.  Default: 0.5*ones(1,n).
%     UPDATES
%               outputFun:          Function.   Default: @sigmoid
%               NOISE:              Function.   Default: @NET_Noise
%               NOISE_MU:           Scalar.     Default 0
%               NOISE_SIGMA:        Scalar.     Default 0
%     ASYMPTOTIC MODE
%               asympMod:           Boolean.    Default: false
%               asymTime:           Scalar.     Default: 100 
%               AM_STEP:            Scalar.     Default 100
%               AM_FP_STEP          Scalar.     Default 4;
%
%               modelType:          String.     Default: 'CTRNN'
%
%   see also PAR_CTRNN;

    net=PAR_CTRNN;
    
    try
        net.externalWMatrix = externalWMatrix;
    catch
        if numExternalInput > 0
            net.externalWMatrix=1-2*rand(numUnits,numExternalInput);
        else
            net.externalWMatrix=0;
        end
    end
    try
        net.internalWMatrix=internalWMatrix;
    catch
        net.internalWMatrix=1-2*rand(numUnits,numUnits);
    end
    try
        net.biases=biases;
    catch
        net.biases=1-2*rand(1,numUnits);
    end

    net.numUnits            = numUnits;
    net.numExternalInput    = size(net.externalWMatrix,2);
    net.externalInput       = zeros(numExternalInput>0,numExternalInput);
    net.initialYValues      = zeros(1,numUnits);
    net.initialOutValues    = 0.5*ones(1,numUnits);
end