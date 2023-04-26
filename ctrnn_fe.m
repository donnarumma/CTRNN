%% Function:     function  [y] = ctrnn_fe (outFun,y0,W,WE,In,theta,tau,steps,deltaT)                                                       
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
%               y_i(s+1)=y_i(s)+(deltaT/tau_i)(-y_i(s)+SUMj(wij*outFun(y_j+theta_j)+SUMj(weij*In_j))                       
%
% LastUpdate    March 28, 2011
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

function  [y] = ctrnn_fe (outFun,y0,W,WE,In,theta,tau,steps,deltaT)

neuronNumber = size(y0,1);

y=zeros(neuronNumber,steps);

ratio = diag (deltaT ./ tau);

f = - y0 + W * feval ( outFun, y0 + theta ) + WE * In;

y(:,1) = y0 + ratio * f;

    for i=1:steps-1
    
        f = - y(:,i) + W * feval ( outFun, y(:,i) + theta ) + WE * In;

        y ( :, (i+1) ) = y (:, i) + ratio * f;

    end;
	
end