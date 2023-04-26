function [o vv] = isFixedEq(val,net)
outFun  = net.outputFun;
eW      = net.externalWMatrix;
iW      = net.internalWMatrix;
biases  = net.biases;
externalInput = net.externalInput;
feval(outFun,val*iW+externalInput*eW+biases)
vv = val-feval(outFun,val*iW+externalInput*eW+biases);
if vv<0.001
    o = 1;
else 
    o = 0;
end
