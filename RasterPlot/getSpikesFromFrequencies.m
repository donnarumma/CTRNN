function spikes=getSpikesFromFrequencies(outs)
%% function spikes=getSpikesFromFrequencies(outs)
% winner take all!
% outs  = TxN frequencies 
% T     = time steps 
% N     = neurons
[T,N]=size(outs);
divs=sum(outs,2);

prob=outs ./ repmat(divs,1,N);

cump=cumsum(prob,2);

r=rand(T,1);

[~,indsp]=max(repmat(r,1,N)<cump,[],2);


spikes=zeros(T,N);
for i=1:T;
    spikes(i,indsp(i))=1;
end
spikes=logical(spikes);
