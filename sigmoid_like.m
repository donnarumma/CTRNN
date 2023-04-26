function y=sigmoid_like(x,min_v,max_v,shift,slope)
if nargin <5
    min_v=0;max_v=65;shift=-2.08;slope=0.452;
end
y=min_v + max_v ./(1+exp(-log10(x+shift))/slope);
end