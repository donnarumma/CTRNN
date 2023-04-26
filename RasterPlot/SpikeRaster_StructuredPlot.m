function out=SpikeRaster_StructuredPlot(spikes,dt)
%% function out=SpikeRaster_StructuredPlot(spikes,dt)

[steps, N]=size(spikes);

S=struct;
S.tvec=0:dt:(dt*steps);

nTrials=1;

for iT=1:nTrials
    for iC=1:N
        S.t{iT,iC}=S.tvec(spikes(:,iC));
        S.label{iT,iC}=['Neuron ' num2str(iC)]; 
    end
end

cfg.freqTheta=8;
cfg.ifmean=0;
cfg.nTrials=nTrials;

PlotSpikeRaster_mod(cfg,S);

if nargin > 0
    out=S;
end    