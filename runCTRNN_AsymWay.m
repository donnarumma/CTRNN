function  [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,externalInput)
%function [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,externalInput)
%INPUT:     net:            Stucture. CTRNN Network parameters
%           externalInput:  nTimes x d matrix. d is the 
%                           number of external input. nTimes is the number
%                           of cycles the network performs. During each
%                           cycle the network input remains unchanged.
%%The network runs in ASYMPTOTIC way.
%
% ASYMPTOTIC WAY
% The network performs nTimes cycles.
% Each cycle ends when the network reaches a maximum 
% number of steps or an 'equilibrium point'.  Note that net.asymTime is
% the time (the maximum number of steps) of each cycle, during each cycle the  
% external input is not changed. Thus, the field net.externalInput is an
% (nTimes x d) matrix. d is the number of external input.
% During the i-th cycle (i=1 to nTimes) the external input
% corresponds to the i-th row of the matrix net.externalInput. The output 
% variable out holds all the network temporal evolution. The evolution of the
% i-th neuron is represented by the i-th columns of out. The output variable  
% outOnlyAsym is a sequence of the nTimes final network  states. 
% Finally, the output variable isFix is a 1xnTimes boolean array where the true
% value in the i-th position suggests that the i-th cycle reached a stable 
% equilibrium point.
% 
% 
% fprintf('Running Asymptotic mode\n');
    nTimes          =size(externalInput,1);
    AM_STEP         =net.AM_STEP; %100;
    AM_FP_STEP      =net.AM_FP_STEP; %4;%10;
    out             =zeros(nTimes+1,net.numUnits);
    out(1,:)        =net.initialOutValues;
    asymTime        =net.asymTime;
    AM_CYCLES       =floor(asymTime/AM_STEP);
    AM_REMAINDER    =rem(asymTime,AM_STEP);
    outOnlyAsym     =zeros(nTimes,net.numUnits);
    isFix           =zeros(1,nTimes);
    iW              =net.internalWMatrix';
    eW              =net.externalWMatrix';
    alfa            =net.dt ./net.tau;
    beta            =1-alfa;
    for n=1:nTimes
        k=size(out,1);
        c=1;
        while c<=AM_CYCLES && isFix(n)==0
            for steps=1:AM_STEP
                k=k+1;
                out(k,:)=beta.*out(k-1,:)+ alfa .*(net.outputFun(out(k-1,:)*iW+externalInput(n,:)*eW+net.biases)+net.NOISE(net));
            end
            isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));
            c=c+1;
        end
        k=size(out,1);
        for steps=1:AM_REMAINDER
            k=k+1;
            out(k,:)=beta .*out(k-1,:)+ alfa .*(net.outputFun(out(k-1,:)*net.internalWMatrix+...
                     externalInput(n,:)*net.externalWMatrix+net.biases)+net.NOISE(net));
        end
        outOnlyAsym(n,:)=out(end,:);
        isFix(n)=isFixedValue(out(end-AM_FP_STEP:end,:));
    end
    out=out(2:end,:); % initial value is not considered. (it is sored in net.initialOutValues)
end
