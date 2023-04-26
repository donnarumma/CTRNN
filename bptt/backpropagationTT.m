function W=backpropagationTT(eta, cicli, intervalli_tempo, ampiezza_intervallo, numero_neuroni)

%costruisco la matrice di pesi iniziale
W=2*eye(numero_neuroni)-2*rand(numero_neuroni);

for k=1:cicli
	
	e=zeros(intervalli_tempo,numero_neuroni);
	for m=1:2
		for n=1:2
	%costruisco il vettore iniziale
	x1=[0.2,0.9];
	x2=[0.2,0.9];
	Y0(1)=sigmoid(x1(m));
	Y0(2)=sigmoid(x2(n));
	for i=3:numero_neuroni
		Y0(i)=0;
	end
		
	%calcolo l'evoluzione del vettore iniziale Y0
	v = sigmoid(W*Y0');
	evoluzione(1,:) = v';
	for i = 2:intervalli_tempo
		v = sigmoid(W*v);
		evoluzione(i,:) = v';
	end
			
	%costruisco la matrice ei
	e(:,3) = e(:,3) + 2*(evoluzione(:,3)-(x1(m)*x2(n)));
	end
	end
			
	%calcolo della z finale e delle derivata dell'errore rispetto ai pesi all'istnte iniziale
	zf = e(intervalli_tempo,:)*ampiezza_intervallo;
	Z(intervalli_tempo,:) = zf;
	for i=1:numero_neuroni
		for j=1:numero_neuroni
			a = sigmoid(W*(evoluzione(intervalli_tempo-1,:))');
			b = 1 - a;
			E(i,j) = zf(i) * evoluzione(intervalli_tempo,j) * a(i) * b(i)*ampiezza_intervallo;
		end
	end
					
	%calcolo la matrice delle derivate della sigma, la matrice delle Z e quella dell'errore
	A=-1.*eye(numero_neuroni);
	for k=0:intervalli_tempo-2
		a = evoluzione(intervalli_tempo-(k+1),:);
		d = (sigmoid(W*a')).*(1-(sigmoid(W*a')));
		for j=1: numero_neuroni
			F(:,j)=A(:,j)+W(:,j).*d;
		end
		c=F*Z(intervalli_tempo-k,:)';
		Z(intervalli_tempo-(k+1),:)=Z(intervalli_tempo-k,:)+ ampiezza_intervallo * (e(intervalli_tempo-(k+1),:)+ c');
        end
        
       for k=0:intervalli_tempo-3
        for i=1:numero_neuroni
	        for j=1:numero_neuroni
		        a=sigmoid(W*(evoluzione(intervalli_tempo-(k+2),:))');
		        b= 1- a;
		        E1(i,j)=Z(intervalli_tempo-(k+1),i) * evoluzione(intervalli_tempo-(k+2),j) * a(i) * b(i);
		end
	end
		
	 E = E+E1*ampiezza_intervallo;
       end
		
       W = W - eta*E;

end
endfunction