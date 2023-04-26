s=load ('sample_outs');
dt=1;
outs=s.outs;
WinnerTakeAll=0;
if WinnerTakeAll
    spikes=getSpikesFromFrequencies(outs);
else
    spikes=getIndependentSpikes(outs);
end
SpikeRaster_StructuredPlot(spikes,dt); 
