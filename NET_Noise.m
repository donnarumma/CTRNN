function   noise=NET_Noise(net)
% function noise=NET_Noise(net)
    
    noise = normrnd(net.NOISE_MU, net.NOISE_SIGMA,1,net.numUnits);
end