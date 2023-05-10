function  [out, isFix, outOnlyAsym]=runCTRNN(net,nTimes)
%function [out, isFix, outOnlyAsym]=runCTRNN(net,nTimes)
%INPUT:     net:        Stucture
%           nTimes:     (Optional) Scalar. Default: nTimes is the number of rows
%                       of the matrix net.externalInput
%The network can run in two differen way: ASYMPTOTIC and NO ASYMPTOTIC way.
%
% ASYMPTOTIC WAY
% If net.asympMod is greater than 0 then  the network runs in asymptotic way. 
% In this case the network performs nTimes cycles.
% Each cycle ends when the network reaches a maximum 
% number of steps or an 'equilibrium point'.  Note that net.asymTime is
% the time (the maximum number of steps) of each cycle, during each cycle the  
% external input is not changed. Thus, the field net.externalInput is an
% (nTimes x d) matrix. During the i-th cycle the external input
% corresponds to the i-th row of the matrix net.externalInput. The output 
% variable out holds all the network temporal evolution. The evolution of the
% i-th neuron is represented by the i-th columns of out. The output variable  
% outOnlyAsym is a sequence of the nTimes final network  states. 
% Finally, the output variable isFix is a 1xnTimes boolean array where the true
% value in the i-th position suggests that the i-th cycle reached a stable 
% equilibrium point.
% 
% 
% NO ASYMPTOTIC WAY
% If net.asympMod is equal to 0 then  the network does not run in asymptotic
% way. In this case the network runs for nTimes steps. In the i-th step
% the external input corresponds to the i-th row of the (nTimes x d) matrix
% net.externalInput.

    numInputTimes   =size(net.externalInput,1);
    asympMod        =net.asympMod;
    externalInput   =net.externalInput;
    
    if nargin < 2
        nTimes=numInputTimes;
    end
    if ~asympMod     
        if numInputTimes==0 
            nTimes=1;
        else
            nTimes=min(nTimes,numInputTimes);
        end    
    else
        if numInputTimes>0 
           nTimes=min(nTimes,numInputTimes);
        end
    end        
        
    outOnlyAsym     =[]; isFix=[];
    if numInputTimes>0 
        if ~asympMod
            [out]=runCTRNN_NoAsymWay(net,externalInput);
        else
            [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,externalInput(1:nTimes,:));
        end
    else
        if ~asympMod           
            net.externalInput=zeros(nTimes,net.numExternalInput);
            [out]=runCTRNN_NoAsymWay(net,externalInput);
        else
            net.externalInput=zeros(1,net.numExternalInput);
            [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,net.externalInput);
        end
    end
end