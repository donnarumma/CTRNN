function net=createCTRNN(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix)
%net=createCTRNN(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix)
%INPUT:         numUnits:           number of neurons (n)
%               numExternalInput:   number of external input (d)
%               biases:             (Optional) 1xn array of biases
%               internalWMatrix:    (Optional) nxn matrix of internal
%                                   weights
%               externalWMatrix:    (Optional) nxd matrix of external
%                                   weights
%
%OUTPUT:        net: It is a structure. The structure fields are:
%               numUnits:           Scalar 
%               numExternalInput:   Scalar
%               biases:             1xd array
%               internalWMatrix:    nxn matrix
%               externalWMatrix:    nxd matrix
%               tau:                1xn array. Default: ones(1,n);
%               outputFun:          String. Default: 'sigmoid'
%               dt:                 Scalar. Default: 0.1
%               asymMod:           Boolean. Default: 0
%               asymTh:            Scalar. Default: 0     
%               externalInput:      Nxd matrix. Default: empty array.
%               initialYValues:     1xn array. Default: zeros(1,n).
%               initialOutValues:   1xn array. Default: 0.5*ones(1,n).
%               asymTime:           Scalar. Default: 100 

    DEFAULT_OUTPUT_FUN='sigmoid';
    DEFAULT_DT=0.1;
    DEFAULT_TAU=1;
    DEFAULT_ASYN_TIME=100;
    net.externalWMatrix=[];
    net.internalWMatrix=[];
    net.biases=[];
    net.modelType = 'CTRNN';
    %% modifica francesco commented = original roberto
%      if nargin <5
%          if numExternalInput >0
%              net.externalWMatrix=1-2*rand(numUnits,numExternalInput);
%          end
%          if nargin < 4
%              net.internalWMatrix=1-2*rand(numUnits,numUnits);
%          else
%              net.internalWMatrix=internalWMatrix;
%          end
%          if nargin <3
%              net.biases=1-2*rand(1,numUnits);
%          else
%              net.biases=biases;
%          end
%      else
%          net.externalWMatrix=externalWMatrix;
%          numExternalInput=size(externalWMatrix,2);
%      end
    if nargin <5
        if numExternalInput >0
            externalWMatrix=1-2*rand(numUnits,numExternalInput);
        end
        if nargin < 4
            internalWMatrix=1-2*rand(numUnits,numUnits);
        end
        if nargin <3
            biases=1-2*rand(1,numUnits);
        end
    else
        numExternalInput	= size(externalWMatrix,2);
    end
    net.externalWMatrix		= externalWMatrix;	% added by francesco
    net.internalWMatrix		= internalWMatrix; 	% added by francesco
    net.biases			= biases;		% added by francesco
    net.numUnits=numUnits;
    net.numExternalInput=numExternalInput;
    net.outputFun=DEFAULT_OUTPUT_FUN;    
    net.tau=DEFAULT_TAU*ones(1,numUnits);
    net.dt=DEFAULT_DT;
    net.asympMod=0;
    net.asymTh=0;
    net.externalInput=[];
%   per roberto: propongo
    net.externalInput = zeros(1,numExternalInput);
%%%%
    net.initialYValues=zeros(1,numUnits);
    net.initialOutValues=0.5*ones(1,numUnits);
    net.asymTime=DEFAULT_ASYN_TIME;
%     net.alfa=net.dt ./net.tau;
%     net.beta= 1-net.alfa;
end