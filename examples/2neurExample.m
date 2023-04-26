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

plot(out_fe,'.r')