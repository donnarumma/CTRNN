TAU_lha_drn=60;
INPUT_ox=[10:100:10^4]';


net=createCTRNN(1,1,0,0,1);
net.outputFun=@sigmoid_like;
net.tau=TAU_lha_drn;
net.dt=0.1*net.tau;
figure;colormap gray;
net.externalInput=INPUT_ox;
net.initialYValues=0;
net.initialOutValues=0;
net.asympMod=1;
[xx,xx,I]=runCTRNN(net);
figure; plot(INPUT_ox,I,'*');
%figure;
% for n=1:length (INPUT_ox)
%     net.externalInput=ones(100,1)*INPUT_ox(n);
%     net.asympMod=0;
%     net.initialYValues=0;
%     net.initialOutValues=0;
%     I2=runCTRNN(net);
%     plot(INPUT_ox(n),I2(end),'*'); hold on;
% end

f= 0.058*((I-0.028 +37.41)>0) .* (I-0.028 +37.41);
figure;plot(INPUT_ox,f,'*');