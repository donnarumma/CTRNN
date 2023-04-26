function spikes=getIndependentSpikes(outs)
randoms=rand(size(outs));
spikes=(randoms<outs);
end