% Simulation of the ctrnn
% two neuron network
% matrix
w11 = -2;
w12 = 2;
w21 = 4;
w22 = -20;
W = [w11,w12; w21, w22];
We = [1,0; ...
      0,1];
% thresholds
theta1 = 0;
theta2 = 0;
theta = [theta1;theta2];

% time constants
tau1 = 1;
tau2 = 1;
tau= [tau1,tau2];

% integration step
deltaT = 0.01;

% iteration 1

% initial condition
y01 = 1;
y02 = 2;
y0 = [y01;y02];
% external input
In1 = 1;
In2 = 1;
In = [In1;In2];

steps1 = 200;
out_fe = [y0 ctrnn_fe('sigmoid',y0, W, We, In, theta, tau, steps1, deltaT)];

% iteration 2
[R C] = size(out_fe);
% initial condition
y01 = out_fe(1,C);
y02 = out_fe(2,C);
y0 = [y01;y02];
% external input
In1 = 25;
In2 = -5;
In = [In1;In2];

steps2 = 200;
out_fe = [out_fe ctrnn_fe('sigmoid',y0, W, We, In, theta, tau, steps2, deltaT)];

% iteration 3
[R C] = size(out_fe);
% initial condition
y01 = out_fe(1,C);
y02 = out_fe(2,C);
y0 = [y01;y02];

% external input
In1 = -2;
In2 = -2;
In = [In1;In2];

steps3 = 800;
out_fe = [out_fe ctrnn_fe('sigmoid',y0, W, We, In, theta, tau, steps3, deltaT)];

% iteration 4
[R C] = size(out_fe);
% initial condition
y01 = out_fe(1,C);
y02 = out_fe(2,C);
y0 = [y01;y02];
% external input
In1 = 5;
In2 = 5;
In = [In1;In2];

steps4 = 300;
out_fe = [out_fe ctrnn_fe('sigmoid',y0, W, We, In, theta, tau, steps4, deltaT)];
 size(out_fe)

%hold off
%plot (0:deltaT:(iter-1)*deltaT,out_fe,"*");
steps =[steps1 steps2 steps3 steps4];
figure
hold on
plot (0:deltaT:(sum(steps))*deltaT,sigmoid(out_fe),'*');

In = [1,1;25,-5;-2,-2;5,5]';
iter = [200,200,800,300];

%plot (iter:deltaT:(iter + (2*iter-1))*deltaT,sigmoid(out_fe),"*");