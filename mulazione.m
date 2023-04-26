function netMul=mulazione(net,matrixType,i,j, minMax,tau)
%netMul=mulazione(net,matrixType,i,j, minMax,tau)
% Sostituisce la connessione w(i,j) con mul. 
% Funzione in due modalità. Se matrixType è uguale a 'internal' allora si
% deve sostituire una connessione tra i nodi della rete, se matrixType è uguale 
% a 'external' allora si deve sostituire una connesione tra un nodo della rete ed
% un input esterno.
% MinMax è un vettore 1x2 che contiene il minimo e il massimo dei pesi della
% rete originale.
% 
% La funzione NON modifica la rete originale (net) ma ne restituisce una 'mulata'.

INTERNAL_MATRIX='internal';
EXTERNAL_MATRIX='external';
MUL_INTERNAL_WMATRIX= [-2.72 -4.13 -11.71;0 0 0;1.12 -1.82 -1.99];
%MUL_INTERNAL_WMATRIX= rand(3,3);
MUL_FIRST_INPUT_WMATRIX=[8.19;0;2.99];
MUL_SECOND_INPUT_WMATRIX=[1.8; 0; -3.69];

netMul=net;
minW=minMax(1);
maxW=minMax(2);
n=size(netMul.internalWMatrix,1);
d=size(netMul.externalWMatrix,2);

%***************************************
netMul.internalWMatrix=[netMul.internalWMatrix zeros(n,3)];
netMul.internalWMatrix=[netMul.internalWMatrix; zeros(3,n+3)];

netMul.externalWMatrix=[netMul.externalWMatrix zeros(n,1)];
netMul.externalWMatrix=[netMul.externalWMatrix; zeros(3,d+1)];
netMul.internalWMatrix(i,n+1)=maxW-minW;
netMul.tau=[netMul.tau ones(1,3)*tau];
netMul.initialOutValues=[netMul.initialOutValues 0.01*ones(1,3)];
netMul.biases=[netMul.biases 0 10 0];

netMul.internalWMatrix(n+1:n+3, n+1:n+3)=MUL_INTERNAL_WMATRIX;  
netMul.externalWMatrix(n+1:n+3,d+1)=MUL_SECOND_INPUT_WMATRIX;

%***************************************
numExtInput=size(netMul.externalInput,1);

if strcmp(matrixType,INTERNAL_MATRIX)==1
    netMul.internalWMatrix(i,j)=minW;
    netMul.internalWMatrix(n+1:n+3,j)=MUL_FIRST_INPUT_WMATRIX; 
    progInput=weights2program(net.internalWMatrix(i,j),minW,maxW);
    netMul.externalInput=[netMul.externalInput repmat(progInput,numExtInput,1)];
elseif strcmp(matrixType,EXTERNAL_MATRIX)==1
    netMul.externalWMatrix(i,j)=minW;
    netMul.externalWMatrix(n+1:n+3,j)=MUL_FIRST_INPUT_WMATRIX;
    progInput=weights2program(net.externalWMatrix(i,j),minW,maxW);
    netMul.externalInput=[netMul.externalInput repmat(progInput,numExtInput,1)];
else
    display('Error: matrixType value can be eiether internal or external');
end
