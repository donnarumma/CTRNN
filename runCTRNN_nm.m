function [out ,net,nm_out_int]=runCTRNN_nm(net,externalInput,nmExternalInput)
%function [out isFix outOnlyAsyn]=runCTRNN(net,nTimes)
% NOTA: DA SCRIVERE!!!!!
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

    numInputTimes=size(externalInput,1);
    net.externalInput=externalInput;
%%  CREAZIONE DELLE VARIABILI NEUROMODULATE
    standardEW=net.stExternalWMatrix;
    standardIW=net.stInternalWMatrix;
    standardB=net.stBiases;
    nmR=net.nmRatio;
    len=floor(numInputTimes/nmR);
    nm_out_alfa_int=[];
    nm_out_alfa_ext=[];
    nm_out_alfa_b=[];
    if not(isempty(net.nm_intW_ind))
        nm_out_int=zeros(len,length(net.nm_intW_ind));
        intParams=net.nmParams.intParams;
    end
    
    if not(isempty(net.nm_extW_ind))
        nm_out_ext=zeros(len,length(net.nm_nm_extW_ind));
    end
    if not(isempty(net.nm_biases_ind))
        nm_out_b=zeros(len,length(net.nm_biases_ind));
    end
    nmRunFun=net.runFun;
    out=zeros(numInputTimes,net.numUnits);
    for i=1:len
        if (net.centerCrossing)
                %extI=net.externalWMatrix * externalInput((i-1)*nmR+1:i*nmR,:)';
                net.biases=-(sum(net.internalWMatrix,2)+sum(net.externalWMatrix,2))/2;
                net.biases=net.biases';
        end
        out((i-1)*nmR+1:i*nmR,:)=runCTRNN_NoAsymWay(net,externalInput((i-1)*nmR+1:i*nmR,:));
        net.initialOutValues=out(i*nmR,:);
        if not(isempty(net.nm_intW_ind))
            [nm_out_int(i,:), intParams]=nmRunFun(intParams,nmExternalInput);
            net.nmParams.intParams=intParams;
            net.internalWMatrix(net.nm_intW_ind)=nm_out_int(i,:);
%             [nm_out_alfa_int(i,:), net.nm_alfaInternalW]=runNM(net.nm_alfaInternalW,nmExternalInput);
%             net.internalWMatrix(net.nm_intW_ind)= (1+nm_out_alfa_int(i,:)) .* standardIW(net.nm_intW_ind);
        end

        if not(isempty(net.nm_extW_ind))
            [nm_out_alfa_ext(i,:), net.nm_alfaExternalW]=runNM(net.nm_alfaExternalW,nmExternalInput);
            net.externalWMatrix(net.nm_extW_ind)= (1+nm_out_alfa_ext(i,:)) .* standardEW(net.nm_extW_ind); 
        end
        if not(isempty(net.nm_biases_ind	))
            [nm_out_alfa_b(i,:), net.nm_alfaBiases]=runNM(net.nm_alfaBiases,nmExternalInput);
            net.biases(net.nm_biases_ind)= (1+nm_out_alfa_b(i,:)) .* standardB(net.nm_biases_ind); 
        end
    end
    

end
% function param=update_nm_parameter(param,nm_out)
%              NM_COEFF=7;
%             if (not(isempty(nm_out)))
%                 param=param+(NM_COEFF*nm_out) ./param;
%             end
% 
% end
% 
function [nm_out, nm]=runNM(nm,nmPerc)
    len=length(nm);
    nm_out  =zeros(1,len);
    for i=1:len
        %nm{i}.externalInput=nmPerc;
        out=runCTRNN_NoAsymWay(nm{i},nmPerc);
        nm_out(i)=out(end,1);
        nm{i}.initialOutValues=nm_out(i);
    end
end

