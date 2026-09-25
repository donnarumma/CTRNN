function p = tva_softmax(logits)
% Stable row-wise softmax for finite real logits.
assert(isreal(logits) && all(isfinite(logits(:))),'tva:Logits','Invalid logits.');
p=zeros(size(logits));
for row=1:size(logits,1)
    shifted=logits(row,:)-max(logits(row,:));
    e=exp(shifted); p(row,:)=e/sum(e);
end
end
