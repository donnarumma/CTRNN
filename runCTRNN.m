function [out, isFix, outOnlyAsym]=runCTRNN(net,nTimes)
%function [out isFix outOnlyAsym]=runCTRNN(net,nTimes)
%INPUT:     net:        Stucture
%           nTimes:     (Optional) Scalar. Default: nTimes is the number of rows
%                       of the matrix net.externalInput
%The network can run in two differen way: ASYMPTOTIC and NO ASYMPTOTIC way.
%
% ASYMPTOTIC WAY
% If net.asymMod is greater than 0 then  the network runs in asymptotic way. 
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
% If net.asymMod is equal to 0 then  the network does not run in asymptotic
% way. In this case the network runs for nTimes steps. In the i-th step
% the external input corresponds to the i-th row of the (nTimes x d) matrix
% net.externalInput.

    numInputTimes=size(net.externalInput,1);
    asymMod=net.asympMod;
    AM_STEP=100;
    AM_FP_STEP=4;%10;
    
    if nargin < 2
        nTimes=numInputTimes;
    end
    if asymMod>0
        
        if numInputTimes==0 
            nTimes=1;
        else
            nTimes=min(nTimes,numInputTimes);
        end
    
    else
        if numInputTimes>0 
           nTimes=min(nTimes,numInputTimes);
        end
        nTimes = nTimes+1; % added by francesco
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
    asymTime=net.asymTime;
    AM_CYCLES=floor(asymTime/AM_STEP);
    AM_REMAINDER=rem(asymTime,AM_STEP);
    outOnlyAsym=[];
    if numInputTimes>0 
        if asymMod==0
            [out]=runCTRNN_NoAsymWay(net,externalInput);
        else
            [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,externalInput(1:nTimes,:));
        end
    else
        if asymMod==0            
            net.externalInput=zeros(nTimes,net.numExternalInput);
            [out]=runCTRNN_NoAsymWay(net,externalInput);
        else
            net.externalInput=zeros(1,net.numExternalInput);
            [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,net.externalInput);
        end
    end
end
%%%%OLD VERSION
% function [out, isFix, outOnlyAsym]=runCTRNN(net,nTimes)
% %function [out isFix outOnlyAsym]=runCTRNN(net,nTimes)
% %INPUT:     net:        Stucture
% %           nTimes:     (Optional) Scalar. Default: nTimes is the number of rows
% %                       of the matrix net.externalInput
% %The network can run in two differen way: ASYMPTOTIC and NO ASYMPTOTIC way.
% %
% % ASYMPTOTIC WAY
% % If net.asymMod is greater than 0 then  the network runs in asymptotic way. 
% % In this case the network performs nTimes cycles.
% % Each cycle ends when the network reaches a maximum 
% % number of steps or an 'equilibrium point'.  Note that net.asymTime is
% % the time (the maximum number of steps) of each cycle, during each cycle the  
% % external input is not changed. Thus, the field net.externalInput is an
% % (nTimes x d) matrix. During the i-th cycle the external input
% % corresponds to the i-th row of the matrix net.externalInput. The output 
% % variable out holds all the network temporal evolution. The evolution of the
% % i-th neuron is represented by the i-th columns of out. The output variable  
% % outOnlyAsym is a sequence of the nTimes final network  states. 
% % Finally, the output variable isFix is a 1xnTimes boolean array where the true
% % value in the i-th position suggests that the i-th cycle reached a stable 
% % equilibrium point.
% % 
% % 
% % NO ASYMPTOTIC WAY
% % If net.asymMod is equal to 0 then  the network does not run in asymptotic
% % way. In this case the network runs for nTimes steps. In the i-th step
% % the external input corresponds to the i-th row of the (nTimes x d) matrix
% % net.externalInput.
% 
%     numInputTimes=size(net.externalInput,1);
%     asymMod=net.asymMod;
%     AM_STEP=100;
%     AM_FP_STEP=4;%10;
%     
%     if nargin < 2
%         nTimes=numInputTimes;
%     end
%     if asymMod>0
%         
%         if numInputTimes==0 
%             nTimes=1;
%         else
%             nTimes=min(nTimes,numInputTimes);
%         end
%     
%     else
%         if numInputTimes>0 
%            nTimes=min(nTimes,numInputTimes);
%         end
%         nTimes = nTimes+1; % added by francesco
%     end        
%     
%     alfa=net.dt ./net.tau;
%     beta= 1-alfa;
%     iW=net.internalWMatrix';
%     eW=net.externalWMatrix';
%     biases=net.biases;
%     outFun=net.outputFun;
%     %y(1,:)=net.initialYValues;
%     externalInput=net.externalInput;
%     out(1,:)=net.initialOutValues;
%     asymTime=net.asymTime;
%     AM_CYCLES=floor(asymTime/AM_STEP);
%     AM_REMAINDER=rem(asymTime,AM_STEP);
%     outOnlyAsym=[];
%     if numInputTimes>0 
%         if asymMod==0
%             for n=2:nTimes
%                 %y(n,:)=y(n-1,:) + alfa .*(-y(n-1,:) + out(n-1,:)*iW+ externalInput(n-1,:)*eW);
%                 %out(n,:)=feval(outFun,y(n,:)+biases);
%                 out(n,:)=beta .*out(n-1,:)+ alfa .*(feval(outFun,out(n-1,:)*iW+externalInput(n-1,:)*eW+biases));
%             end
%             out = out(2:end,:); % added by francesco
%         else
%             outOnlyAsym=zeros(nTimes,size(iW,1));
%             for n=1:nTimes
%                 isFix(n)=0;
%             %%extInput=net.externalInput(n,:)
%                 k=size(out,1);
%                 c=1;
%                 %AM_CYCLES
%                 while c<=AM_CYCLES && isFix(n)==0
%                     
%                     for steps=1:AM_STEP
%                         k=k+1;
%                         out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+externalInput(n,:)*eW+biases));
%                     end
%                     isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));
%                     %isFix(n)=isFixedEq(out(end-AM_FP_STEP:end,:));
% 
%                     c=c+1;
%                 end
%                 k=size(out,1);
%                 for steps=1:AM_REMAINDER
%                     k=k+1;
%                     out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+externalInput(n,:)*eW+biases));
%                 end
%                 outOnlyAsym(n,:)=out(end,:);
%                 isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));        
%             end
%         end
%     else
%         if asymMod==0
%             for n=2:nTimes
%                 %y(n,:)=y(n-1,:) + alfa .*(-y(n-1,:) + out(n-1,:)*iW);
%                 %out(n,:)=feval(outFun,y(n,:)+biases);
%                 out(n,:)=beta .*out(n-1,:)+ alfa .*(feval(outFun,out(n-1,:)*iW+biases));
%             end
%             out = out(2:end,:); % added by francesco
%         else
%            outOnlyAsym=zeros(nTimes,size(iW,1));
%            for n=1:nTimes
%                 isFix(n)=0;
%             %%extInput=net.externalInput(n,:)
%                 k=size(out,1);
%                 c=1;
%                 while c<=AM_CYCLES && isFix(n)==0
%                     
%                     for steps=1:AM_STEP
%                         k=k+1;
%                         out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+biases));
%                     end
%                     isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));
%                     c=c+1;
%                 end
%                 k=size(out,1);
%                 for steps=1:AM_REMAINDER
%                     k+1;
%                     out(k,:)=beta .*out(k-1,:)+ alfa .*(feval(outFun,out(k-1,:)*iW+biases));
%                 end
%                 outOnlyAsym(n,:)=out(end,:);
%                 isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));        
%            end
%         end
%     end
% end
