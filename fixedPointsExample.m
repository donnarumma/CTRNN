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
INTERNAL_MAT=[7 1;-1 7];
%EXTERNAL WEIGHT MATRIX (DEFAULT VALUES)
%EXTERNAL_MAT=[4.72 0;0 5.83];
EXTERNAL_MAT=[0 0;0 0];
%BIASES VALUES (DEFAULT VALUES)
%centerCrossing
BIASES=-((sum(INTERNAL_MAT,2)+sum(EXTERNAL_MAT,2))/2)';

%%
net=createCTRNN(NUM_NEURONS,NUM_NEURONS);
net.internalWMatrix=INTERNAL_MAT;
%net.internalWMatrix=[3 1;-1 3];
%net.externalWMatrix=[-2 0;0 -1];
net.externalWMatrix=EXTERNAL_MAT;
net.biases=BIASES;
net.tau=[1 1];

net.asymMod=1;
%net.asynTime=1000; 
net.asymTime=1000; 
net.externalInput=[1 1];
%f1=figure;
%f2=figure;
figure;
count_i=0; 
for i=0:0.1:1
    count_i=count_i+1;
    count_j=0;
    for j=0:0.1:1
        count_j=count_j+1;
        net.initialOutValues=[i j];

        [out]=runCTRNN_NoAsymWay(net,zeros(net.asymTime,2));
        plot(out(:,1), out(:,2),'k') 
        hold on;
        totalOut{count_i,count_j}=out;
    end
end
xlabel('y_1 (spike rate)');ylabel('y_2 (spike rate)');
colormap gray;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=totalOut{1,1};

%figure;
%plot(out(:,1),out(:,2))

spikes=getIndependentSpikes(out(100:10:end,:));
figure;
SpikeRaster_StructuredPlot(spikes,net.dt); 

out=totalOut{count_i,count_j};
colormap gray;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=totalOut{1,10};

%figure;
%plot(out(:,1),out(:,2))

spikes=getIndependentSpikes(out(100:10:end,:));
figure;
SpikeRaster_StructuredPlot(spikes,net.dt); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=totalOut{10,1};

%figure;
%plot(out(:,1),out(:,2))

spikes=getIndependentSpikes(out(100:10:end,:));
figure;
SpikeRaster_StructuredPlot(spikes,net.dt); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=totalOut{10,10};
colormap gray;
%figure;
%plot(out(:,1),out(:,2))

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
