function [out isFix outOnlyAsyn]=runCTRNNPrevete(net,nTimes)
%function [out isFix outOnlyAsyn]=runCTRNN(net,nTimes)
%INPUT:     net:        Stucture
%           nTimes:     (Optional) Scalar. Default: nTimes is the number of rows
%                       of the matrix net.externalInput
%The network can run in two differen way: ASYMPTOTIC and NO ASYMPTOTIC way.
%
% ASYMPTOTIC WAY
% If net.asynMod is greater than 0 then  the network runs in asymptotic way. 
% In this case the network performs nTimes cycles.
% Each cycle ends when the network reaches a maximum 
% number of steps or an 'equilibrium point'.  Note that net.asynTime is
% the time (the maximum number of steps) of each cycle, during each cycle the  
% external input is not changed. Thus, the field net.externalInput is an
% (nTimes x d) matrix. During the i-th cycle the external input
% corresponds to the i-th row of the matrix net.externalInput. The output 
% variable out holds all the network temporal evolution. The evolution of the
% i-th neuron is represented by the i-th columns of out. The output variable  
% outOnlyAsyn is a sequence of the nTimes final network  states. 
% Finally, the output variable isFix is a 1xnTimes boolean array where the true
% value in the i-th position suggests that the i-th cycle reached a stable 
% equilibrium point.
% 
% 
% NO ASYMPTOTIC WAY
% If net.asynMod is equal to 0 then  the network does not run in asymptotic
% way. In this case the network runs for nTimes steps. In the i-th step
% the external input corresponds to the i-th row of the (nTimes x d) matrix
% net.externalInput.

    numInputTimes=size(net.externalInput,1);
    asyntMod=net.asyntMod;
    AM_STEP=100;
    AM_FP_STEP=4;%10;
    
    if nargin < 2
        nTimes=numInputTimes;
    end
    if asyntMod>0
        
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
    
    alfa=net.dt ./net.tau;
    beta= 1-alfa;
    iW=net.internalWMatrix';
    eW=net.externalWMatrix';
    biases=net.biases;
    outFun=net.outputFun;
    %y(1,:)=net.initialYValues;
    externalInput=net.externalInput;
    out(1,:)=net.initialOutValues;
    asynTime=net.asynTime;
    AM_CYCLES=floor(asynTime/AM_STEP);
    AM_REMAINDER=rem(asynTime,AM_STEP);
    outOnlyAsyn=[];
    if numInputTimes>0 
        if asyntMod==0
            for n=2:nTimes
                %y(n,:)=y(n-1,:) + alfa .*(-y(n-1,:) + out(n-1,:)*iW+ externalInput(n-1,:)*eW);
                %out(n,:)=feval(outFun,y(n,:)+biases);
                out(n,:)=beta .*out(n-1,:)+ alfa .*(feval(outFun,out(n-1,:)*iW+externalInput(n-1,:)*eW+biases));
            end
        else
            outOnlyAsyn=zeros(nTimes,size(iW,1));
            for n=1:nTimes
                isFix(n)=0;
            %%extInput=net.externalInput(n,:)
                k=size(out,1);
                c=1;
                %AM_CYCLES
                while c<=AM_CYCLES && isFix(n)==0
                    
                    for steps=1:AM_STEP
                        k=k+1;
                        out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+externalInput(n,:)*eW+biases));
                    end
                    isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));
                    c=c+1;
                end
                k=size(out,1);
                for steps=1:AM_REMAINDER
                    k=k+1;
                    out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+externalInput(n,:)*eW+biases));
                end
                outOnlyAsyn(n,:)=out(end,:);
                isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));        
            end
        end
    else
        if asyntMod==0
            for n=2:nTimes
                %y(n,:)=y(n-1,:) + alfa .*(-y(n-1,:) + out(n-1,:)*iW);
                %out(n,:)=feval(outFun,y(n,:)+biases);
                out(n,:)=beta .*out(n-1,:)+ alfa .*(feval(outFun,out(n-1,:)*iW+biases));
            end
        else
           outOnlyAsyn=zeros(nTimes,size(iW,1));
           for n=1:nTimes
                isFix(n)=0;
            %%extInput=net.externalInput(n,:)
                k=size(out,1);
                c=1;
                while c<=AM_CYCLES && isFix(n)==0
                    
                    for steps=1:AM_STEP
                        k=k+1;
                        out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+biases));
                    end
                    isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));
                    c=c+1;
                end
                k=size(out,1);
                for steps=1:AM_REMAINDER
                    k+1;
                    out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+biases));
                end
                outOnlyAsyn(n,:)=out(end,:);
                isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));        
           end
        end
    end
end
