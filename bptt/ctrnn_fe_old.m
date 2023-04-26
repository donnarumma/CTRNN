%% CTRNN forward euler approximation
% [y] = ctrnn_fe (outFun,y0,W,In,theta,tau,time, deltaT)
% outFun: the name of the activation function to use (ex.'sigmoid')
% y0: column vector of the initial condition of the network
%     [y0_1; ... ;y0_n]
% W: matrix of the weights [w11, w12, ..., w1n; 
%                           w21, w22, ..., w2n; 
%                           ...  ...  ...  ...; 
%                           wn1  ...       wnn] 
% theta: column vector of the thresholds [theta_1; ...; [theta_n]
% tau; vector of the time constants [tau_1, ... , tau_n]
% steps: steps of integration
% deltaT: integration step
% output= matrix n x steps [y_1(0), y_1(1), ..., y_1(steps);                             
% Equations solved:
% y_i(s+1)=y_i(s)+(deltaT/tau_i)(-y_i(s)+SUMj(wij*outFun(y_j+theta_j))+In_j)

function  [y] = ctrnn_fe (outFun,y0,W,WE,In,theta,tau,steps,deltaT);
%iter = time/deltaT;

[R C] = size(y0);
y=zeros(R,steps);
[RE CE]=size(WE);
In0=zeros(CE,1);

ratio = diag (deltaT ./ tau);

f = - y0 +  W  * feval(outFun,y0 + theta) + WE * In0;
y(:,1)= y0 + ratio * f;

for i=1:steps-1 
  f =  - y(:,i) +  W  * feval(outFun,y(:,i) + theta) + WE * In(:,i);

  y(:,(i+1))= y(:,i) + ratio * f;

end
	
end