function net=createCTRNN_nm(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix,nmElements,runFun,nmParams)
%function net=createCTRNN_nm(numUnits,numExternalInput,biases,internalWMatrix,externalWMatrix,nmElements,nmNoise)
%INPUT:         numUnits:           number of neurons (n)
%               numExternalInput:   number of external input (d)
%               biases:             (Optional) 1xn array of biases
%               internalWMatrix:    (Optional) nxn matrix of internal
%                                   weights
%               externalWMatrix:    (Optional) nxd matrix of external
%                                   weights
%               nmElements:       (Optional) Cell array reporting the
%                                   neuromodulated weights and biases.
%                                   The first element is a boolean nxn matrix.
%                                   The second element is a boolean nxd matrix.
%                                   The third element is a boolean 1xn
%                                   matrix. True value corresponds to a
%                                   neuromodulated value.
%OUTPUT:        net: It is a structure. The structure fields are:
%               numUnits:           Scalar 
%               numExternalInput:   Scalar
%               biases:             1xd array
%               internalWMatrix:    nxn matrix
%               externalWMatrix:    nxd matrix
%               tau:                1xn array. Default: ones(1,n);
%               outputFun:          String. Default: 'sigmoid'
%               dt:                 Scalar. Default: 0.1
%               asympMod:           Boolean. Default: 0
%               asympTh:            Scalar. Default: 0     
%               externalInput:      Nxd matrix. Default: empty array.
%               initialYValues:     1xn array. Default: zeros(1,n).
%               initialOutValues:   1xn array. Default: 0.5*ones(1,n).
%               asympTime:           Scalar. Default: 100 

    DEFAULT_OUTPUT_FUN='sigmoid';
    DEFAULT_DT=0.1;
    DEFAULT_TAU=1;
    DEFAULT_ASYM_TIME=1000;
    DEFAULT_NM_RATIO=10;
    net.externalWMatrix=[];
    net.internalWMatrix=[];
    net.biases=[];
    net.modelType = 'CTRNN';
    net.centerCrossing=false;
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
    
    if nargin<6
        nmElements={false(numUnits,numUnits), false(numUnits,numExternalInput), false(1,numUnits)};
    end
    if nargin <7
        nmNoise=0;
    end
    net.nmElements=nmElements;
    net.externalWMatrix		= externalWMatrix;	% added by francesco
    net.internalWMatrix		= internalWMatrix; 	% added by francesco
    net.biases			= biases;		% added by francesco
    net.numUnits=numUnits;
    net.numExternalInput=numExternalInput;
    net.outputFun=DEFAULT_OUTPUT_FUN;    
    net.tau=DEFAULT_TAU*ones(1,numUnits);
    net.dt=DEFAULT_DT;
    net.asympMod=0;
    net.asympTh=0;
    net.externalInput=[];
%   per roberto: propongo
    net.externalInput = zeros(1,numExternalInput);
%%%%
    net.initialYValues=zeros(1,numUnits);
    net.initialOutValues=0.5*ones(1,numUnits);
    net.asympTime=DEFAULT_ASYM_TIME;
    net.nmRatio=DEFAULT_NM_RATIO;
    net.stInternalWMatrix=net.internalWMatrix;
    net.stExternalWMatrix=net.externalWMatrix;
    net.stBiases=net.biases;
%    net.initFun=initFun;
    net.runFun = runFun;
    net.nm_intW_ind = nm_indFun(net.nmElements{1});
    net.nm_extW_ind = nm_indFun(net.nmElements{2});
    net.nm_biases_ind = nm_indFun(net.nmElements{3});
    %     [net.nm_alfaInternalW,net.nm_intW_ind]=init_nm_var(net.nmElements{1},0,net.tau,net.dt,nmNoise);
%     [net.nm_alfaExternalW,net.nm_extW_ind]=init_nm_var(net.nmElements{2},0,net.tau,net.dt,nmNoise);
%     [net.nm_alfaBiases,net.nm_biases_ind]=init_nm_var(net.nmElements{3},0,net.tau,net.dt,nmNoise);
    net.nmParams = nmParams;
end

function nm_ind=nm_indFun(nm_params)
    nm_ind=find(nm_params==true);
    if iscolumn(nm_ind)
        nm_ind=nm_ind';
    end
    
end

% function [nm_var,nm_ind]=init_nm_var(nm_params,nmPerc,tau,dt,noise)
%     nm_ind=find(nm_params==true);
%     if iscolumn(nm_ind)
%         nm_ind=nm_ind';
%     end
%     BIAS=-10;
%     W11=10;
%     W_EXT=10;
%     if noise>0
%         BIAS=BIAS+noise*randn;
%         W11=W11+noise*randn;
%         W_EXT=W_EXT+noise*randn;
%     end
%     nm_var=cell(0);
%     for i=1:length(nm_ind)
%         nm_var{i}=createCTRNN(1,1);
%         nm_var{i}.internalWMatrix=W11;
%         nm_var{i}.externalWMatrix=W_EXT;
%         nm_var{i}.biases=BIAS;
%         nm_var{i}.tau=tau(1);
%         nm_var{i}.initialOutValues=[0];
%         nm_var{i}.asyntMod=0;
%         nm_var{i}.asynTime=2500; 
%         nm_var{i}.externalInput=nmPerc;
%         nm_var{i}.dt=dt(1);
%     end
% end