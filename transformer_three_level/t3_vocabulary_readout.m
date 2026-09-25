function result = t3_vocabulary_readout(Y, Wreadout, breadout, vocabulary)
%T3_VOCABULARY_READOUT Illustrative untrained linear + row-softmax head.
% Y is the token-by-feature block output. The four-symbol default vocabulary
% is only a set of labels: this function is not a trained language model.
assert(size(Y,2)==size(Wreadout,1) && size(Wreadout,2)==numel(breadout), ...
    't3:ReadoutShape','Readout parameter shapes disagree.');
assert(iscell(vocabulary) && numel(vocabulary)==size(Wreadout,2), ...
    't3:Vocabulary','One label is required per output symbol.');
logits=t3_programmed_linear(Y,Wreadout,'right')+repmat(reshape(breadout,1,[]),size(Y,1),1);
assert(all(isfinite(logits(:))),'t3:ReadoutFinite','Readout logits must be finite.');
probabilities=zeros(size(logits));
for row=1:size(logits,1)
    offset=max(logits(row,:)); denominator=0;
    for column=1:size(logits,2)
        probabilities(row,column)=exp(logits(row,column)-offset);
        denominator=denominator+probabilities(row,column);
    end
    for column=1:size(logits,2)
        probabilities(row,column)=probabilities(row,column)/denominator;
    end
end
[~,indices]=max(probabilities,[],2);
result=struct('vocabulary',{reshape(vocabulary,1,[])},'logits',logits, ...
    'probabilities',probabilities,'prediction_indices',indices, ...
    'predicted_symbols',{reshape(vocabulary(indices),[],1)});
end
