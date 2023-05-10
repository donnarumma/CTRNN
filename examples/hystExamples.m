close all;
NUM_POINTS=50;

%alpha=7;
%alpha=4.5;
alpha=5;

m1=3;
%m1=1;
%D1=0.1;
D1=0.1;
%c1=0.2;
c1=0.2;
u=linspace(0,1,NUM_POINTS);
%u=[0:0.5:1];
y0=0;
%y1=hysteresis(u,y0,alpha,m1,D1,c1);figure;plot(u,y1,'r*');

%y=(hysteresis((2*u-1),0.6*y0 -0.3,alpha,m1,D1,c1)+0.3)./0.6;figure;plot(u,y,'r*');
y=(hysteresis((2*u-1)-0.3,0.6*y0 -0.3,alpha,m1,D1,c1)+0.3)./0.6;figure;plot(u,y,'r*');

hold on;

% u=linspace(u(end),u(end),20);
% %u=[0:0.5:1];
% y0=y(end);
% y=(hysteresis((2*u-1),0.6*y0 -0.3,alpha,m1,D1,c1)+0.3)./0.6;plot(u,y,'gx');
% hold on;


u=linspace(u(end),0,NUM_POINTS);
%u=[u ones(1,100)*0.4];
y0=y(end);
y=(hysteresis((2*u-1),0.6*y0 -0.3, alpha,m1,D1,c1)+0.3)./0.6;plot(u,y,'ko');

% u=[[-1:0.1:0.5] [0.5:-0.1:-1]];
% y=(hysteresis(u,-0.3,5,3,0.1,00.6)+0.3)./0.6;figure;plot(u,y)
