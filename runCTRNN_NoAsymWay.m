function [out]=runCTRNN_NoAsymWay(net,externalInput)
%function [out]=runCTRNN_NoAsymWay(net,externalInput)
%INPUT:     net:            Stucture
%           externalInput:  nTimes x d matrix. d is the 
%                           number of external input. nTimes is the number
%                           of steps the network performs. 
%The network runs in NO ASYMPTOTIC way.
%
% 
% NO ASYMPTOTIC WAY
%The network does not run in asymptotic
% way. In this case the network runs for nTimes steps. In the i-th step
% the external input corresponds to the i-th row of the (nTimes x d) matrix
% externalInput.

    nTimes=size(externalInput,1);
    alfa=net.dt ./net.tau;
    beta= 1-alfa;
    iW=net.internalWMatrix';
    eW=net.externalWMatrix';
    biases=net.biases;
    outFun=net.outputFun;
    out(1,:)=net.initialOutValues;
    for n=2:nTimes+1
        %y(n,:)=y(n-1,:) + alfa .*(-y(n-1,:) + out(n-1,:)*iW+ externalInput(n-1,:)*eW);
        %out(n,:)=feval(outFun,y(n,:)+biases);
        out(n,:)=beta .*out(n-1,:)+ alfa .*(feval(outFun,out(n-1,:)*iW+externalInput(n-1,:)*eW+biases));
    end
    out = out(2:end,:); % added by francesco
end
