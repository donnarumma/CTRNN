%% backpropagationTT5
%Questa funzione addestra una rete CTRNN ad associare ad una sequenza di input un certo output  con un algoritmo di backpropagationTT che e' un algoritmo di discesa del gradiente. Piu' precisamente la rete viene addestrata ad associare una determinata sequenza di input, uno stato della rete in cui l'output del terzo neurone sia assegnato dall'utente.
%L'algorimo usa come modello di CTRNN  Y(:,t) = Y(:,t-1) * (1-ampiezza_intervallo) + ampiezza_intervallo*W*sigmoid(Y(:,t-1))+ampiezza_intervallo*WE*matrice_input(:,t).
%
%[W,WE,errore]=backpropagationTT5(eta, cicli, intervalli_tempo, ampiezza_intervallo, numero_neuroni,Y0,target,matrice_input,numero_input)
% eta: fattore che moltipica il valore del gradiente della funzione d'errore rispetto ai pesi.
%cicli: numero delle ere. Numero delle volte in cui viene calcolato il valore del gradiente della funzione di errore e vengono aggiornati i pesi. 
%intervalli_tempo: numero di istanti in cui si decide si seguire il comportamento della rete neurale.
%ampiezza_intervallo: misura dell'ampiezza del singolo intervallo di tempo. 
%tau: tempo caratteristico dei neuroni
%numero_neuroni: numero di neuroni della rete.
%Y0: condizione iniziale di tutti i neuroni della rete.
%target: vettore riga i cui elementi sono i valori che il terzo neurone della rete deve cercare di assumere durante il tempo.
%matrice_input: ad ogni colonna deve essere associato l'input dato alla rete a qull'istante di tempo.
%numero_input: numero degli elementi di input, questo numero è uguale al numero delle colonne della matrice_input.



function [W, WE, errore, n] = backpropagationTTnew ( net , params )
	if isfield(params,'repetitions')
		n = params.repetitions;
	else
		n = 1;
	end
	weightsUpdate		= params.learningAlgorithm;

	cicli 			= params.maxEpochs;
% 	eta             = params.eta; 
	matrice_input 		= params.trainSet';
	matrice_target		= params.trainTarget';
	intervalli_tempo	= size(params.trainSet,1);
	
	ampiezza_intervallo 	= net.dt;
	numero_neuroni		= net.numUnits;
	numero_input		= net.numExternalInput;
	tau 			= net.tau;

	%costruisco la matrice iniziale dei pesi
%  	[numero_input,intervalli_tempo] = size(matrice_input);
%  	f		= matrice_input(:,1);
%  	g		= matrice_input(:,2);

	%calcolo quante volte lo stesso valore viene ripetuto in input alla rete
%  	n=1;
%  	while f==g
%  		n=n+1;
%  		f=matrice_input(:,n);
%  		g=matrice_input(:,n+1);
%  	end
	%le matrici W e WE vengono inizializzate con numeri casuali compresi nell'intervallo [-10,10]
	W 		= net.internalWMatrix;
	WE		= net.externalWMatrix;
	Y0		= net.initialYValues';
	net.Ys(1,:)	= net.initialYValues;
	for l=1:cicli
		
		ETFW	= zeros(numero_neuroni);
		ETFWE	= zeros(numero_neuroni,numero_input);
		EW	= zeros(numero_neuroni);
		EWE	= zeros(numero_neuroni,numero_input);
		errore	= 0;
		
		net.indexOut		= 1;
		net.internalWMatrix 	= W;
		net.externalWMatrix 	= WE;
		for kk = 1:intervalli_tempo
			net.externalInput = params.trainSet(kk,:);
			net = ctrnn_fe_oneStep(net);
		end
		
		Y = (net.Ys(2:end,:))';
		Ym(:,1) = Y0.*(1-ampiezza_intervallo./tau)' + (ampiezza_intervallo./tau)'.*(W*sigmoid(Y0)+WE*matrice_input(:,1));
		
		%calcolo l'evoluzione del k-esimo elemento del training set
		for t = 2:intervalli_tempo
			Ym(:,t) = Ym(:,t-1) .* (1-ampiezza_intervallo./tau)' + (ampiezza_intervallo./tau)'.*(W*sigmoid(Ym(:,t-1))+WE*matrice_input(:,t-1));
		end
% 		sum( sum ( abs (Ym - Y) ) )
% 		Ym(end) - Y(end)
        Y = Ym;
		if mod(l,100)==0
			figure;
%  			plot(Y(3,:));
			plot(sigmoid(Y(3,:)));
			hold on
			plot(params.trainTarget);
			hold off;
		end
		%calcolo la matrice delle e=dE'/dyi, all'istante finale. Dove E = E'(t1)*Dt+...+E'(tf)*Dt
		e=zeros(numero_neuroni,intervalli_tempo);
		
		for k=0:(intervalli_tempo/n-1)
			for t=1:n
				e(3,n*k+t)= 2*(sigmoid(Y(3,n*k+t))-matrice_target(n*k+t))*sigmoid(Y(3,n*k+t))*(1-sigmoid(Y(3,n*k+t)))*(t/n);
				errore1=(sigmoid(Y(3,n*k+t))-matrice_target(n*k+t))^2*(t/n);
				errore=errore1+errore;
			end
		end % endk
			
		Z=zeros(numero_neuroni,intervalli_tempo);
			
		%Calcolo la Z all'istante finale
		Z(:,intervalli_tempo)=e(:,intervalli_tempo)*ampiezza_intervallo;
			
			
		%calcolo la matrice della derivata dell'errore rispetto ai pesi all'istante finale
		for i=1:numero_neuroni
			for j=1:numero_neuroni
				ETFW(i,j)=Z(i,intervalli_tempo)*sigmoid(Y(j,intervalli_tempo-1)).*(ampiezza_intervallo./tau(j));
			end % endj
		end % endi
		

		for i=1:numero_neuroni
			for j=1:numero_input
				ETFWE(i,j)=Z(i,intervalli_tempo)*matrice_input(j,intervalli_tempo)*ampiezza_intervallo/tau(j);
			end % endj
		end % endi
					
		%inizio a calcolare le Z e le derivate per tutti i tempi
					
		for m=1:intervalli_tempo-1
			for i=1:numero_neuroni
				a=Y(i,intervalli_tempo-m);
				b=sigmoid(a);
				c=(1-b);
				d=0;
						
				for h=1:numero_neuroni
					d1=Z(h,intervalli_tempo-m+1)*W(h,i)*c*b;
					d=d+d1;
				end
				Z(i,intervalli_tempo-m)=(1-ampiezza_intervallo/tau(i))*Z(i,intervalli_tempo-m+1) + ampiezza_intervallo/tau(i) * (e(i,intervalli_tempo-m)+d);
			end
		end
		
		
		%calcolo la matrice della derivata dell'errore rispetto ai pesi per tutti i tempi                       
		for i=1:numero_neuroni
			for j=1:numero_neuroni
				for t=1:intervalli_tempo-2
					E1W(i,j,t+1) = Z(i,t+1)*sigmoid(Y(j,t))*ampiezza_intervallo/tau(j);
					EW(i,j) = EW(i,j)+E1W(i,j,t+1);
				end
				EW(i,j) = EW(i,j)+Z(i,1)*sigmoid(Y0(j))*ampiezza_intervallo/tau(j);
			end
		end
		
		
		for i=1:numero_neuroni
			for j=1:numero_input
				for t=1:intervalli_tempo-2
					E1WE(i,j,t+1) = Z(i,t+1)*matrice_input(j,t)*ampiezza_intervallo/tau(i);
					EWE(i,j) = EWE(i,j)+E1WE(i,j,t+1);
				end
				EWE(i,j) = EWE(i,j)+Z(i,1)*matrice_input(1,t)*ampiezza_intervallo/tau(i);
			end
		end
		

		%stampo l'errore e aggiorno le matrici
		fprintf('Iter: %d/%d |Error: %20.20g\n',l,params.maxEpochs,errore);

		[Wtotal] = weightsUpdate({[W WE]},{[EW + ETFW, EWE + ETFWE]}, params);
%  		Wold  	= W  - eta*(EW +ETFW );
%  		WEold 	= WE - eta*(EWE+ETFWE);
		W 	= Wtotal{1} (1:net.numUnits,1:net.numUnits);
		WE 	= Wtotal{1} (1:net.numUnits,net.numUnits+1:net.numUnits+net.numExternalInput); 
%  		sum(sum(abs(Wold-W))) + sum(sum(abs(WEold-WE)))
	end % endl
end % end backpropagationTT	
