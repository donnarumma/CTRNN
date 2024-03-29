function PlotSpikeRaster_mod(cfg_in,S)
% function PlotSpikeRaster(cfg,S)
%
%   INPUTS:
%       cfg_in: input cfg
%       S: input ts NOTE: S.t can be a MxN cell array where M is the number of repeated trials
%       and N is the number of labeled signals
%
%   CFG OPTIONS:
%       cfg.SpikeHeight - default 0.4
%           Size of marker for each spike
%
%       cfg.lfp - default NONE
%           lfp data as a tsd
%
%       cfg.axisflag - default 'tight'
%           Converts axis scaling. TIGHT restricts the plot to the data range. ALL sets
%           the x-axis to cover the whole experiment.
%
%       cfg.Color - default 'k' (black)
%           Color specifier for spikes in a single trial. Can be any MATLAB string color
%           specifier. NOTE: multiple trials will call upon a function to create a good
%           colormap to distinguish cells.
%
%
% MvdM 2014-07-20
% youkitan 2014-11-06 edit
fs=12;
[nTrials,nCells] = size(S.t);
%% Set cfg parameters
cfg_def.SpikeHeight = 0.4;
cfg_def.axisflag = 'tight';
cfg_def.Color = zeros(nCells,3);
%cfg_def.Color = linspecer(nCells);

cfg = ProcessConfig2(cfg_def,cfg_in);
    
try
    xlab=cfg.xlab;
catch
    xlab='time [s]';
end
try
    newfigure=cfg.newfigure;
catch
    newfigure=0;
end
if newfigure
    figure;
end

try
    trialsToPlot=cfg_in.trialsToPlot;
catch
    trialsToPlot=1:nTrials;
end
nTrials=length(trialsToPlot);
try
    th=cfg.th;
catch
    th=0;
end
cmap = cfg.Color;

ddy=0;

freqTheta=cfg_in.freqTheta;
dtfreq=1/freqTheta;
t = 0:dtfreq:max(S.tvec);

%% Plot spikes
debugmode=0;
%Check number of runs (trials) and cells
lw  = 2;
ifmean=cfg.ifmean;
if  nTrials == 1
    hold on; grid on; box on;
    iT=trialsToPlot;
    for iC = 1:nCells
        if isempty(S.t{iT,iC})
            continue;
        end
%         h{iC} = plot([S.t{iC} S.t{iC}],[iC-cfg.SpikeHeight iC+cfg.SpikeHeight],cfg.Color);
        if debugmode
            fprintf('PlotSpikeRaster_mod: Warning - h{iC} = plot([S.t{iC} S.t{iC}],[iC-cfg.SpikeHeight iC+cfg.SpikeHeight],cfg.Color);\n')
        end
        %         x_temp=[S.t{iC}' S.t{iC}'];
        x_temp=S.t{iT,iC};
        x_val=[];
        if ifmean
            count=0;
            for i=1:length(t)-1
                if  t(i)>min(S.tvec)
                    count=count+1;
%                     fprintf('Interval [%g, %g]\n',t(i),t(i+1));
                    goods= x_temp>t(i) & x_temp<t(i+1);
                    
                    if sum(goods)>th
                        x_val(1,count)=mean(x_temp(goods));
                    else
                        x_val(1,count)=NaN;
                    end
                end
            end
        else
            x_val=x_temp;
        end
%         x_val
%         fprintf ('Plotting Neuron%g, x_val=%g\n',iC,x_val);
%         x_val
%         pause
        x_val=[x_val' x_val'];
        
        if length(x_temp)==2
            h{iC} = plot([x_temp(1),x_temp(1)],[iC-cfg.SpikeHeight iC+cfg.SpikeHeight],'Color',cmap(iC,:),'linewidth',lw);
            h{iC} = plot([x_temp(2),x_temp(2)],[iC-cfg.SpikeHeight iC+cfg.SpikeHeight],'Color',cmap(iC,:),'linewidth',lw);
            %fprintf('PlotSpikeRaster_mod: Generic plot([S.t{iT,iC} S.t{iT,iC}],[iT-cfg.SpikeHeight iT+cfg.SpikeHeight]);\n')
        else
            h{iC} = plot(x_val,[iC-cfg.SpikeHeight iC+cfg.SpikeHeight],'Color',cmap(iC,:),'linewidth',lw);
        end
    end
    
    % Set axis labels
    ylabel('Cell #','fontsize',fs);
    set(gca,'Ytick',1:nCells);

    set(gca,'YtickLabel',S.label(iT,:));

    % Set ylims
    ylims= [1 nCells];

    
else %equivalent to elseif nTrials > 1 (nTrials is always a positive integer)
    
    hold on; grid on; box on;
    for iC = 1:nCells
        for iT = trialsToPlot
            if length(S.t{iT,iC})==2
                x_temp=S.t{iT,iC};
                plot([x_temp(1),x_temp(1)],[iT-cfg.SpikeHeight iT+cfg.SpikeHeight],'Color',cmap(iC,:),'linewidth',lw);
                plot([x_temp(2),x_temp(2)],[iT-cfg.SpikeHeight iT+cfg.SpikeHeight],'Color',cmap(iC,:),'linewidth',lw);
%                fprintf('PlotSpikeRaster_mod: Generic plot([S.t{iT,iC} S.t{iT,iC}],[iT-cfg.SpikeHeight iT+cfg.SpikeHeight]);\n')
            else 
%                fprintf('Cell: %g, Trial: %g, size: %g\n',iC,iT,length(S.t{iT,iC}));
                if S.t{iT,iC}>0
                    plot([S.t{iT,iC}' S.t{iT,iC}'],[iT-cfg.SpikeHeight iT+cfg.SpikeHeight],'Color',cmap(iC,:),'linewidth',lw);
                end
            end
        end %iterate trials
    end %iterate cells
    
    % Set axis labels
    ylabel('Trial','fontsize',fs);

    set(gca,'Ytick',trialsToPlot);
%    set(gca,'YtickLabel',num2cell(cfg.trialsToPlot));
    % Set ylims
    ylims = [min(trialsToPlot) max(trialsToPlot)];

end

xlabel(xlab,'fontsize',fs);

%% Plot LFP
% return
if isfield(cfg,'lfp')
    
    spikelims = get(gca,'Xlim');
%     figure;
    % Make sure lfp data has same time interval as spiking data
    if (min(cfg.lfp.tvec) < spikelims(1) || max(cfg.lfp.tvec) > spikelims(2))...
            && strcmp(cfg.axisflag,'tight')
        fprintf('minlfp = %d maxlfp = %d minspikes = %d maxspikes = %d\n',...
            min(cfg.lfp.tvec),max(cfg.lfp.tvec),spikelims(1),spikelims(2))
        fprintf('Range of lfp data exceeds spiketrain data!... Restricting lfp data...');
        cfg.lfp = restrict(cfg.lfp,spikelims(1),spikelims(2));
    end
    ddy=-5;
    cfg.lfp.data = rescale(cfg.lfp.data,ddy,1);
    plot(cfg.lfp.tvec,cfg.lfp.data,'Color',0.8*[1 1 1]);
end


%% Adjust figure

% Set Axis limits
xlims = get(gca,'Xlim');
dy=0.6;%1

switch cfg.axisflag
    case 'tight'
        xlim([xlims(1) xlims(2)]);
        ylim([ylims(1)-dy ylims(2)+dy]);
    case 'all'
        xlim([S.cfg.ExpKeys.TimeOnTrack S.cfg.ExpKeys.TimeOffTrack]);
        ylim([ylims(1)-dy ylims(2)+dy])
end

% ylim([1-dy+ddy nCells+dy]);
hold off;

end