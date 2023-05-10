clear all;
close all;
%NUMBERS OF NEURONS;
NUM_NEURONS=3;
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
INTERNAL_MAT=[4.5 1 0;-1.2 4.5 1;0 -1 4.5];
%EXTERNAL WEIGHT MATRIX (DEFAULT VALUES)
EXTERNAL_MAT=[4.72 0 0;0 5.83 0;0 0 5.83];
%BIASES VALUES (DEFAULT VALUES)
BIASES=[-7 -7 -7];

%%
net=createCTRNN(NUM_NEURONS,NUM_NEURONS);
net.internalWMatrix=INTERNAL_MAT;
%net.internalWMatrix=[3 1;-1 3];
%net.externalWMatrix=[-2 0;0 -1];
net.externalWMatrix=EXTERNAL_MAT;
net.biases=BIASES;
net.tau=[10 10 10];

net.asyntMod=1;
%net.asynTime=1000; 
net.asynTime=10000; 
net.externalInput=[0.9 0.9 0.9];
% f1=figure;
% f2=figure;
% for i=0:0.2:1
%     for j=0:0.2:1
% 
%         net.initialOutValues=[i j];
% 
%         [out]=runCTRNN(net);
%         figure(f1);
%         hold on;
%         plot(out(:,1), 'r') 
%         figure(f2);
%         hold on;
%         plot(out(:,2), 'g') 
%         
%         
%     end
% end


figure;
for i=0.0:0.4:1
    for j=0.0:0.4:1
        for k=0.0:0.4:1
            net.initialOutValues=[i j k];

            [out]=runCTRNN(net);
            plot3(out(:,1),out(:,2),out(:,3)) 
            hold on;
        end
    end
end

figure;
plot(out(:,1),'r');
hold on;
plot(out(:,2),'g');
hold on;
plot(out(:,3),'b');

net.internalWMatrix(1,1)=net.internalWMatrix(1,1)+1;
net.internalWMatrix(2,2)=net.internalWMatrix(2,2)+1;
net.internalWMatrix(3,3)=net.internalWMatrix(3,3)+1;
figure;
for i=0.0:0.4:1
    for j=0.0:0.4:1
        for k=0.0:0.4:1
            net.initialOutValues=[i j k];

            [out]=runCTRNN(net);
            plot3(out(:,1),out(:,2),out(:,3)) 
            hold on;
        end
    end
end

figure;
plot(out(:,1),'r');
hold on;
plot(out(:,2),'g');
hold on;
plot(out(:,3),'b');

