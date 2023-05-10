function y=isFixedValue(out)
TH=10^(-6);
dist=0;
len=size(out,1);
count=1;
while ((dist<TH) && (count<len))
    dist=norm(out(count+1,:)-out(count,:));
    count=count +1;
end
if count==len
    y=1;
else
    y=0;
end
end
