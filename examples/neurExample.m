%  % Simulation of the ctrnn
%  % two neuron network
%  % matrix
%  w11 = -2;
%  w12 = 2;
%  w21 = 4;
%  w22 = -20;
%  W = [w11,w12; w21, w22];
%  We = [1,0; ...
%        0,1];
%  % thresholds
%  theta1 = 0;
%  theta2 = 0;
%  theta = [theta1;theta2];
%  
%  % time constants
%  tau1 = 1;
%  tau2 = 1;
%  tau= [tau1,tau2];
%  
%  % integration step
%  deltaT = 0.01;
%  
%  % iteration 1
%  
%  % initial condition
%  y01 = 1;
%  y02 = 2;
%  y0 = [y01;y02];
%  % external input
%  In1 = 1;
%  In2 = 1;
%  In = [In1;In2];
%  
%  steps1 = 200;
%  
%  out_fe = ([y0 ctrnn_fe('sigmoid',y0, W, We, In, theta, tau, steps1, deltaT)]);
%  
%  out_dual_fe = ([y0 ctrnn_dual_fe('sigmoid', sigmoid(y0), W, We, In, theta, tau, steps1, deltaT)]);
%  
%  close all; figure; 
%  hold on; grid on;
%  plot((out_fe(1,:)),out_fe(2,:),'.r');
%  out_dual_fe = W * out_dual_fe + repmat(In,1,length(out_dual_fe));
%  plot((out_dual_fe(1,:)),out_dual_fe(2,:),'.k');

% Simulation of the ctrnn
% one neuron network
close all; clear all;
% matrix
W = 10;
We = 1;
% thresholds
theta = 0;

% time constants
tau= 1;

% integration step
deltaT = 0.01;

% iteration 1

% initial condition
x0 = 1;
% external input
In = 2;

steps = 200;

y0 = W * x0 + We*In;

y  = ([y0      ctrnn_fe('sigmoid', y0, W, We, In, theta, tau, steps, deltaT)]);

x  = ([x0 ctrnn_dual_fe('sigmoid', x0, W, We, In, theta, tau, steps, deltaT)]);


net			= createCTRNN(1,1);
net.internalWMatrix	= W;
net.externalWMatrix	= We;
net.biases		= theta;
net.tau			= tau;
net.initialOutValues	= x0;	
net.asyntMod		= 0;
net.asynTime		= steps;
net.dt 			= deltaT;
net.externalInput	= repmat(In',steps+1,1);
%  [x]			= runCTRNN(net)';

figure; 
hold on; grid on;
plot(y,'.r');
new_y = W * x + repmat(We*In,1,length(x));
plot(new_y,'.k');

