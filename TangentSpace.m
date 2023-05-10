function   tx=TangentSpace(net,tp)
% function tx=TangentSpace(net,tp)
iW      =net.internalWMatrix';
eW      =net.externalWMatrix';
biases  =net.biases;
outFun  =net.outputFun;
EI      =net.externalInput(1,:);

tx =(-tp+outFun(tp*iW+EI*eW+biases)+net.NOISE(net) ) ./ net.tau;