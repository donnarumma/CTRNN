function spikes=getIndependentSpikesWithNoise(outs,noise)
try
    noise;
catch
    noise=0.1;
    noise=0.05;
end

randomsFire=rand(size(outs));
randomsOff =rand(size(outs));

randoms=rand(size(outs));

spikes=(randoms<outs);
spikes(randomsOff<noise) =0;
spikes(randomsFire<noise)=1;
end
