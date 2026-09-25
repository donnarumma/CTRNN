function r = tva_readout(z, p)
% Token-wise postnorm block and decision head; class probabilities are NOT z.
assert(size(z,2)==3,'tva:ReadoutShape','Readout requires three features.');
xq=ones(size(z));
r.projected=tva_linear(z,p.Wout,'right');
r.residual1=xq+r.projected;
r.U=layernorm(r.residual1,p.gamma1,p.beta1,p.layernorm_epsilon);
r.hidden=max(0,tva_linear(r.U,p.W1,'right')+repmat(p.b1,size(z,1),1));
r.ffn=tva_linear(r.hidden,p.W2,'right')+repmat(p.b2,size(z,1),1);
r.residual2=r.U+r.ffn;
r.Y=layernorm(r.residual2,p.gamma2,p.beta2,p.layernorm_epsilon);
r.logits=tva_linear(r.Y,p.Wclass,'right')+repmat(p.bclass,size(z,1),1);
r.pclass=tva_softmax(r.logits);
end

function y=layernorm(x,gamma,beta,epsilon)
y=zeros(size(x));
for i=1:size(x,1)
    mu=sum(x(i,:))/size(x,2);
    variance=sum((x(i,:)-mu).^2)/size(x,2);
    y(i,:)=((x(i,:)-mu)/sqrt(variance+epsilon)).*gamma+beta;
end
end
