%% Function:     function  [y] = ctrnn_dual_fe (outFun,y0,W,WE,In,theta,tau,steps,deltaT)                                                       
%                  
% Authors:      Francesco Donnarumma, Guglielmo Montone, Roberto Prevete
%               Dipartimento di Scienze Fisiche, 
%               Universita' di Napoli Federico II
%
% Description:  CTRNN forward euler approximation
%          
% Parameters:   outFun: the name of the activation function to use (e.g.'sigmoid')
%               y0: column vector nx1 of the initial condition of the network [y0_1; ... ;y0_n]
%               W : nxn matrix of the connection weights [w11, w12, ..., w1n; 
%                                                         w21, w22, ..., w2n; 
%                                                         ...  ...  ...  ...; 
%                                                         wn1  ...  ...  wnn] 
%               We: nxp matrix of the input weights [we11, we12, ..., we1p; 
%                                                    we21, we22, ..., we2p; 
%                                                     ...   ...  ...   ...; 
%                                                    wenp  ...   ...  wenp] 
%               In = vector px1 of the inputs, one for each neuron [I1;I2;...;Ip]
%               theta: column vector nx1 of the thresholds [theta_1; ...; theta_n]
%               tau; vector of the n time constants [tau_1, ... , tau_n]
%               steps: steps of integration
%               deltaT: integration step
% Return value: output= matrix n x steps [y_1(1), y_1(2), ..., y_1(steps);
%                                         y_2(1), y_2(2), ..., y_2(steps);
%                                            ...     ...  ...,        ...;
%                                         y_n(1), y_n(2), ..., y_n(steps)]
%               Equations solved:
%               y_i(s+1)=y_i(s)+(deltaT/tau_i)(-y_i(s)+outFun(SUMj(wij*y_j)+theta_j+SUMj(weij*In_j)))                       
%
% LastUpdate    January 02, 2012
%
% Note:
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 1, or (at your option)
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details. A copy of the GNU 
% General Public License can be obtained from the 
% Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

%INPUT:         numUnits:           number of neurons (n)
%               numExternalInput:   number of external input (d)
%               biases:             (Optional) 1xn array of biases
%               internalWMatrix:    (Optional) nxn matrix of internal
%                                   weights
%               externalWMatrix:    (Optional) nxd matrix of external
%                                   weights
%
%OUTPUT:        net: It is a structure. The structure fields are:
%               numUnits:           Scalar 
%               numExternalInput:   Scalar
%               biases:             1xd array
%               internalWMatrix:    nxn matrix
%               externalWMatrix:    nxd matrix
%               tau:                1xn array. Default: ones(1,n);
%               outputFun:          String. Default: 'sigmoid'
%               dt:                 Scalar. Default: 0.1
%               asyntMod:           Boolean. Default: 0
%               asyntTh:            Scalar. Default: 0     
%               externalInput:      Nxd matrix. Default: empty array.
%               initialYValues:     1xn array. Default: zeros(1,n).
%               initialOutValues:   1xn array. Default: 0.5*ones(1,n).
%               asynTime:           Scalar. Default: 100 


function [net] = ctrnn_fe_oneStep (net)%outFun,x0,W,WE,In,theta,tau,steps,deltaT)
%  	outFun		= net.outputFun;
%  	x0 		= net.initialOutValues;
%  	W		= net.internalWMatrix;
%  	WE		= net.externalWMatrix;
%  	In		= net.externalInput;
%  	biases		= net.biases;
%  	tau 		= net.tau;
%  	deltaT		= net.dt;
%  	numUnits 	= net.numUnits;
	outFun		= str2func(net.outputFun);
    
%      alfa	= net.dt ./net.tau;
%      beta	= 1-alfa;    
%  out(n,:)=beta .*out(n-1,:)+ alfa .*(feval(outFun,out(n-1,:)*iW+externalInput(n-1,:)*eW+biases));
%  	size(net.externalWMatrix.')
%  	size(net.externalInput * net.externalWMatrix.')
	ratio		=   net.dt ./ net.tau;
	f		= - net.Ys(net.indexOut,:) + outFun ( net.Ys(net.indexOut,:) + net.biases ) * net.internalWMatrix.' + net.externalInput * net.externalWMatrix.';
	y		=   net.Ys(net.indexOut,:) + ratio .* f;
	net.indexOut 	=   net.indexOut+1;
	net.Ys(net.indexOut,:) = y;
end