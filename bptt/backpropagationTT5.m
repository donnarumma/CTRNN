%% backpropagationTT5
%Questa funzione addestra una rete CTRNN ad associare ad una sequenza di input un certo output  
%con un algoritmo di backpropagationTT che e' un algoritmo di discesa del gradiente. Piu' precisamente
%la rete viene addestrata ad associare una determinata sequenza di input, uno stato della rete in cui 
%l'output del terzo neurone sia assegnato dall'utente.
%L'algorimo usa come modello di CTRNN:  
% 
% Y(:,t) = Y(:,t-1) * (1-ampiezza_intervallo) + ampiezza_intervallo*W*sigmoid(Y(:,t-1))+ampiezza_intervallo*WE*input_train(:,t).
%
%[W,WE,errore]=backpropagationTT5(eta, cicli, intervalli_tempo, ampiezza_intervallo, numero_neuroni,Y0,target,input_train,numero_input)
% eta: fattore che moltipica il valore del gradiente della funzione d'errore rispetto ai pesi.
%cicli: numero delle ere. Numero delle volte in cui viene calcolato il valore del gradiente della funzione di errore e vengono aggiornati i pesi.
%intervalli_tempo: numero di istanti in cui si decide si seguire il comportamento della rete neurale.
%ampiezza_intervallo: misura dell'ampiezza del singolo intervallo di tempo.
%tau: tempo caratteristico dei neuroni
%numero_neuroni: numero di neuroni della rete.
%Y0: condizione iniziale di tutti i neuroni della rete.
%target: vettore riga i cui elementi sono i valori che il terzo neurone della rete deve cercare di assumere durante il tempo.
%input_train: ad ogni colonna deve essere associato l'input dato alla rete a qull'istante di tempo.
%numero_input: numero degli elementi di input, questo numero è uguale al numero delle colonne della input_train.



function net = backpropagationTT5(net,learnParams)

tau                 = net.tau;
ampiezza_intervallo = net.dt;
numero_neuroni      = net.numUnits;
intervalli_tempo    = net.timeStep;
ActFun              = net.outputFun;
numero_input        = net.numExternalInput;
Y0                  = net.initialYValues;
theta               = net.biases;

derivActFun         = learnParams.derivActFunctions;
input_train         = learnParams.trainSet;
target_train        = learnParams.trainTarget;
input_validation    = learnParams.validSet;
target_validation   = learnParams.validTarget;
cicli               = learnParams.maxEpochs;
eta                 = learnParams.eta;
stopCriteria        = learnParams.stopCriteria;
stopIterToll        = learnParams.stopIterToll;



%le matrici W e WE vengono inizializzate con numeri casuali compresi nell'intervallo [-10,10]
W=10*(ones(numero_neuroni) - 2*rand(numero_neuroni));
WE=10*(ones(numero_neuroni,numero_input)-2*rand(numero_neuroni,numero_input));

E_V=zeros(1,cicli);
E_T=zeros(1,cicli);

num_target_elem = size(input_train,2);
num_validation_elem = size(input_validation,2);

l = 1;

while (l<=cicli && ~stopCriteria(l,E_V,stopIterToll))
%for l=1:cicli
    
    ETFW=zeros(numero_neuroni);
    ETFWE=zeros(numero_neuroni,numero_input);
    EW=zeros(numero_neuroni);
    EWE=zeros(numero_neuroni,numero_input);
    
    Y = cell(1,num_target_elem);
    for i = 1:num_target_elem
        Y{1,i} = ctrnn_fe (ActFun,Y0',W,WE,input_train(:,i),theta',tau,intervalli_tempo,ampiezza_intervallo);
    end
    
    YV = cell(1,num_validation_elem);
    for i = 1:num_validation_elem
        YV{i} = ctrnn_fe (ActFun,Y0',W,WE,input_validation(:,i),theta',tau,intervalli_tempo,ampiezza_intervallo);
    end
    %Y(:,1) = Y0*(1-ampiezza_intervallo/tau) + ampiezza_intervallo/tau*(W*sigmoid(Y0)+WE*input_train(:,1));
    
    %calcolo l'evoluzione del k-esimo elemento del training set
    %for t=2:intervalli_tempo
    %    Y(:,t) = Y(:,t-1) * (1-ampiezza_intervallo/tau) + ampiezza_intervallo/tau*(W*sigmoid(Y(:,t-1))+WE*input_train(:,t-1));
    %end
    
    %calcolo la matrice delle e=dE'/dyi, all'istante finale. Dove E = E'(t1)*Dt+...+E'(tf)*Dt
    e=zeros(numero_neuroni,intervalli_tempo*num_target_elem);
    
    for k=1:num_target_elem
        for t=1:intervalli_tempo
            
            index = intervalli_tempo*(k-1)+t;
            
            e(:,index)= 2*(sigmoid(Y{k}(:,t))-target_train(:,k)).*sigmoid(Y{k}(:,t)).*(1-sigmoid(Y{k}(:,t))).*(t/intervalli_tempo);
            
            errore_target=norm(sigmoid(Y{k}(:,t))-target_train(:,k)).*(t/intervalli_tempo);
            E_T(l) = errore_target+E_T(l);
            
        end
    end
    
    for k = 1:num_validation_elem
        for t = 1:intervalli_tempo
            errore_validation=norm((sigmoid(YV{k}(:,t))-target_validation(:,k))).*(t/intervalli_tempo);
            E_V(l) = errore_validation+E_V(l);
        end
    end
    
    
    Z=zeros(numero_neuroni,intervalli_tempo*num_target_elem);
    
    %Calcolo la Z all'istante finale
    Z(:,intervalli_tempo*num_target_elem)=e(:,intervalli_tempo*num_target_elem)*ampiezza_intervallo;
    
    
    
    %calcolo la matroce della derivata dell'errore rispetto ai pesi all'istante finale
    
    %forse va un for su k piuttosto che num_target_elem
    for i=1:numero_neuroni
        for j=1:numero_neuroni
            ETFW(i,j)=Z(i,intervalli_tempo*num_target_elem)*sigmoid(Y{num_target_elem}(j,intervalli_tempo-1))*ampiezza_intervallo/tau(1);
        end
    end
    
    
    for i=1:numero_neuroni
        for j=1:numero_input
            ETFWE(i,j)=Z(i,intervalli_tempo*num_target_elem)*input_train(j,num_target_elem)*ampiezza_intervallo/tau(1);
        end
    end
    
    %inizio a calcolare le Z e le derivate per tutti i tempi
    top = intervalli_tempo*num_target_elem;
    
    for m= 1:num_target_elem
        for t=1:intervalli_tempo
            for i=1:numero_neuroni
                a=Y{num_target_elem-m+1}(i,intervalli_tempo-t+1);
                b=sigmoid(a);
                c=(1-b);
                d=0;
                index = num_target_elem*(m-1) + t;
                
                for h=1:numero_neuroni
                    d1=Z(h,top-index+1)*W(h,i)*c*b;
                    d=d+d1;
                end
                
                Z(i,top-index)=(1-ampiezza_intervallo/tau(1))*Z(i,top-index+1) + ampiezza_intervallo/tau(1) * (e(i,top-index)+d);
            end
        end
    end
    
    
    %calcolo la matroce della derivata dell'errore rispetto ai pesi per tutti i tempi
    for i=1:numero_neuroni
        for j=1:numero_neuroni
            for k = 1:num_target_elem
                for t=1:intervalli_tempo-2
                    E1W(i,j,intervalli_tempo*(k-1)+t+1) = Z(i,intervalli_tempo*(k-1)+t+1)*sigmoid(Y{k}(j,t))*ampiezza_intervallo/tau(1);
                    EW(i,j) = EW(i,j)+E1W(i,j,intervalli_tempo*(k-1)+t+1);
                end
                EW(i,j) = EW(i,j)+Z(i,1)*sigmoid(Y0(j))*ampiezza_intervallo/tau(1);
                
            end
        end
    end
    
    
    for i=1:numero_neuroni
        for j=1:numero_input
            for k = 1:num_target_elem
                for t=1:intervalli_tempo-2
                    E1WE(i,j,intervalli_tempo*(k-1)+t+1) = Z(i,intervalli_tempo*(k-1)+t+1)*input_train(j,k)*ampiezza_intervallo/tau(1);
                    EWE(i,j) = EWE(i,j)+E1WE(i,j,intervalli_tempo*(k-1)+t+1);
                end
                EWE(i,j) = EWE(i,j)+Z(i,1)*input_train(1,k)*ampiezza_intervallo/tau(1);
            end
        end
    end
    
    
    %stampo l'errore e aggiorno le matrici
    fprintf('Epochs %4d/%d | E_T: %20.10g | E_V: %6.3f\n',l,E_T(l), E_V(l));
    
    
    W=W - eta*(EW+ETFW);
    WE=WE - eta*(EWE+ETFWE);
    
    l = l+1;
end

net.internalWMatrx  = W;
net.externalWMatrix = WE;
