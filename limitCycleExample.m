close all; clear all;
addpath('./RasterPlot');
%NUMBERS OF NEURONS;
NUM_NEURONS=2;
%NUMBER OF INPUT
NUM_INPUT=2;
%INTERNAL WEIGHT MATRIX (DEFAULT VALUES)
INTERNAL_MAT=2 .*rand(NUM_NEURONS,NUM_NEURONS)-1;
%EXTERNAL WEIGHT MATRIX (DEFAULT VALUES)
EXTERNAL_MAT=2 .*rand(NUM_NEURONS,NUM_INPUT)-1;
%BIASES VALUES (DEFAULT VALUES)
BIASES=2 .*rand(1,NUM_NEURONS)-1;

%% USER DEFINED VALUES
%INTERNAL WEIGHT MATRIX (DEFAULT VALUES)
INTERNAL_MAT=[4.5 1;-1 4.5];
%EXTERNAL WEIGHT MATRIX (DEFAULT VALUES)
EXTERNAL_MAT=[4.72 0;0 5.83];
%BIASES VALUES (DEFAULT VALUES)
BIASES=[-7 -7];

%%
net=createCTRNN(NUM_NEURONS,NUM_NEURONS);
net.internalWMatrix=INTERNAL_MAT;
%net.internalWMatrix=[3 1;-1 3];
%net.externalWMatrix=[-2 0;0 -1];
net.externalWMatrix=EXTERNAL_MAT;
net.biases=BIASES;
net.tau=[1 1];

net.asymMod=0;
%net.asynTime=1000; 
net.asymTime=2000; 
net.externalInput=[0.9 0.9];
%f1=figure;
%f2=figure;
figure;
for i=0:0.2:1
    for j=0:0.2:1

        net.initialOutValues=[i j];

        [out]=runCTRNN(net);
        plot(out(:,1), out(:,2),'k') 
        hold on;
    end
end
colormap gray;
figure;
plot(out(:,1))

spikes=getIndependentSpikes(out(100:10:end,:));
figure;
SpikeRaster_StructuredPlot(spikes,net.dt); 


% for delta_b=-4:2:4
%     net.biases(1)=BIASES(1)+delta_b;
%     net.biases(2)=BIASES(2)+delta_b;
%     for delta_w=-4:2:4
%         %net.biases(1)=b;
%         net.internalWMatrix(1,1)=INTERNAL_MAT(1,1)+delta_w;
%         net.internalWMatrix(2,2)=INTERNAL_MAT(2,2)+delta_w;
%         figure;
%         for i=0:0.1:1
%             for j=0:0.1:1
% 
%                 net.initialOutValues=[i j];
% 
%                 [out]=runCTRNN(net);
%                 plot(out(:,1), out(:,2)) 
%                 hold on;
%             end
%         end
%         title(['Phase portrait, bias 1:' num2str(net.biases(1)) 'bias 2:' num2str(net.biases(2)) 'ac1:' num2str(net.internalWMatrix(1,1)) 'ac2:' num2str(net.internalWMatrix(2,2))]);
%     end
% end
