function   [TP,TX]=computeTangentSpaceGrid(net,GRID_LIMS,N_GRID)
% function [TP,TX]=computeTangentSpaceGrid(net,GRID_LIMS,N_GRID)

xProbe          = linspace(GRID_LIMS(net.Obs(1),1),GRID_LIMS(net.Obs(1),2),N_GRID);
yProbe          = linspace(GRID_LIMS(net.Obs(2),1),GRID_LIMS(net.Obs(2),2),N_GRID);

[XP,YP]         = meshgrid(xProbe, yProbe);
TP              = [XP(:),YP(:)];
TX              = net.TangentSpace(net,TP);