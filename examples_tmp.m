close all;
clear all;
NumNetworks=20;
NOISE_NM=0.4;
h_phasePortrait=plotBifMapCenterCross2neurons;
h_w=figure;
PLOT_FIGURES=0;
for r=1:NumNetworks
    
%NUMBERS OF NEURONS;
NUM_NEURONS=2;
%NUMBER OF INPUT
NUM_INPUT=1;
%INTERNAL WEIGHT MATRIX (DEFAULT VALUES)
w11=4.2+rand*1.1;
w22=3.6+rand*1.2;
figure(h_phasePortrait);
plot(w11,w22,'r*');
INTERNAL_MAT=[w11 1;-1 w22];
%EXTERNAL WEIGHT MATRIX (DEFAULT VALUES)
%EXTERNAL_MAT=[4.72 0;0 5.83];
EXTERNAL_MAT=[0;0];
%BIASES VALUES (DEFAULT VALUES)
BIASES=[-(INTERNAL_MAT(1,1)+INTERNAL_MAT(1,2))/2 -(INTERNAL_MAT(2,2)+INTERNAL_MAT(2,1))/2];
%BIASES=[-3.617 -1.72];
nmParams.intParams.p_standard =[w11 w22]';

%PARAMETRI CHE MI QUANTIFICANO, in percentuale, LO SPOSTAMENTO DEI PESI (in questo caso:w11 e w22) quando
%risulatano neuromodulati. Il valore betaMat(i,j) indica la variazione 
%da applicare al parametro i pesata da quanto e' neuromodulato dal neuromodulatore j.
nmParams.intParams.betaMat = 0.5*rand(2,2);

%PARAMETRI CHE MI DICONO DA quali neuromodulatori un un parametro è influenzato.
%Il valore alphaMat(i,j) indica se
%parametro i e' neuromodulato dal neuromodulatore j. Un valore pari a 0
%indica nessuna influenza. Da notare i parametri in questione sono già
%quelli che potenzialmente possono essere neuromodulati (in questo caso w11 e w22).
nmParams.intParams.alphaMat = zeros(2,2);
nmParams.intParams.alphaMat(1,1)=1;nmParams.intParams.alphaMat(2,2)=1;
% nmParams.intParams.th1      = 0.4+0.02*randn(2,2);
% nmParams.intParams.th2      = nmParams.intParams.th1 +0.2+0.02*randn(2,2);
    
%net=createCTRNN_nm(NUM_NEURONS,NUM_INPUT,BIASES,INTERNAL_MAT,EXTERNAL_MAT,{[true false;false true], [], []},@nmRule,nmParams);


%E' creata una rete con 2 parametri neuromodulati, vedi:{[true false;false true], [], []}
% e con regola di neuromodulazione @hyst_nmRule,nmParams
net=createCTRNN_nm(NUM_NEURONS,NUM_INPUT,BIASES,INTERNAL_MAT,EXTERNAL_MAT,{[true false;false true], [], []},@hyst_nmRule,nmParams);

net.tau=10*ones(1,NUM_NEURONS);
net.asympTime=800;
net.centerCrossing=true;
net.initialOutValues=[0 0];
% [out, f, isFix, outOnlyAsym]=plotPhasePortrait(net,0);
% hold on;
% title(['w11:' num2str(w11) ' w22:' num2str(w22)]);
externalInput=zeros(2000,1);
%for I=1
c=1;
I_old=0;

%E' lanciata una rete con 2 parametri neuromodulati da due neuromodulatori
rangeI=0:0.05:1;
for I=rangeI %QUI E' DA VERIFICARE SE POSSO METTERE QUALUNQUE STEP
    [~,net,nm_out_int{c}]=runCTRNN_nm(net,externalInput,[linspace(I_old, I,5);linspace(I_old, I,5)]);
%     [out, f, isFix, outOnlyAsym]=plotPhasePortrait(net,0);
%     hold on;
%     title(['I: ' num2str(I)]);
%    pp(r,c)=isFix;
%     net.nmParams.intParams.alphaMat
%     net.nmParams.intParams.betaMat
%     if isfield(net.nmParams.intParams,'nmStateAlphaMat')
%         net.nmParams.intParams.nmStateAlphaMat
%     end
    ww(c,:)=[net.internalWMatrix(1,1),net.internalWMatrix(2,2)];
    nm{c}=net.nmParams.intParams.nmStateAlphaMat;
    c=c+1;
    I_old=I;
end
figure(h_w);
hold on;
plot(rangeI,ww(:,1),'r*');
hold on;
plot(rangeI,ww(:,2),'bo');
c=c-1;
figure(h_phasePortrait);
plot(net.internalWMatrix(1,1),net.internalWMatrix(2,2),'b*');
quiver(w11,w22,net.internalWMatrix(1,1)-w11,net.internalWMatrix(2,2)-w22,0);
% for I=1:-0.2:0 %QUI E' DA VERIFICARE SE POSSO METTERE QUALUNQUE STEP
%     [~,net,nm_out_int_2{c}]=runCTRNN_nm(net,externalInput,[linspace(I_old, I,5);linspace(I_old, I,5)]);
%     [out, f, isFix, outOnlyAsym]=plotPhasePortrait(net,0);
%     hold on;    title(['I: ' num2str(I)]);
%     nn(r,c)=isFix;
%     c=c-1;
%     I_old=I;
%     figure(h_phasePortrait);
%     plot(net.internalWMatrix(1,1),net.internalWMatrix(2,2),'y*');
% 
% end
%close all;
end
