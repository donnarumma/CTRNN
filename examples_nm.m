%NUMBERS OF NEURONS;
NUM_NEURONS=2;
%NUMBER OF INPUT
NUM_INPUT=2;
%INTERNAL WEIGHT MATRIX (DEFAULT VALUES)
INTERNAL_MAT=[4.5 1;-1 4.5];
%EXTERNAL WEIGHT MATRIX (DEFAULT VALUES)
EXTERNAL_MAT=[4.72 0;0 5.83];
%BIASES VALUES (DEFAULT VALUES)
BIASES=[-7 -7];


%%
%numExternalInput,biases,internalWMatrix,externalWMatrix
net=createCTRNN_nm(NUM_NEURONS,NUM_INPUT,BIASES,INTERNAL_MAT,EXTERNAL_MAT,{[1 0;0 1] [] []});
% net.internalWMatrix=INTERNAL_MAT;
% net.externalWMatrix=EXTERNAL_MAT;
% net.biases=BIASES;
net.tau=10*ones(1,NUM_NEURONS);

net.asyntMod=1;
%net.asynTime=1000; 
net.asynTime=5000; 
net.externalInput=[0.9 0.9];
figure;

for i=0:0.2:1
    for j=0:0.2:1
        net.initialOutValues=[i j];
        %net.nmInput=1;
        %[out,~,~,net]=runCTRNN(net);
        [out]=runCTRNN(net);
        plot(out(:,1),out(:,2));
        hold on;
    end
end
title('Initial Behaviour');
net.initialOutValues=out(end,:);
for nmPercent=0.2:0.2:1
    net.nmInput=nmPercent;
    [out,~,~,net]=runCTRNN_nm(net);
    figure;
    for i=0:0.2:1
        for j=0:0.2:1
            net.initialOutValues=[i j];
            %net.nmInput=1;
            %[out,~,~,net]=runCTRNN(net);
            [out]=runCTRNN(net);
            plot(out(:,1),out(:,2));
            hold on;
        end
    end
    title(['Behaviour with I=' num2str(nmPercent) ' w11:' num2str(net.internalWMatrix(1,1)) ' w22:' num2str(net.internalWMatrix(2,2))]);
end