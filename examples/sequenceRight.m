function [out] = sequenceRight (duration,	...
				sigma_duration)
if nargin < 1
  duration = 200;
end
if nargin < 2
  sigma_duration = 50;
end

outFun	= 'sigmoid';
y0	=  0;
W	=  -20;
We 	=   1;
theta	=   0;
deltaT	=   0.2;
tau	=   1;
steps	= 10;

sp 		=   1;
threshUp	=  15;
threshDown	=   5;

up 		=  25;
down 		= - 5;


lenseq = round(duration + randn*sigma_duration);

Inup = rand (1,lenseq)*(threshUp - threshDown) + threshDown;
fino = threshUp + 3 + abs(randn)*3;
Inup = [Inup , Inup(end):sp:fino, fino:-sp:10];

%  lenseq = round(200 + randn*50);

%  Indown = rand (1,lenseq)*(up - threshDown) + threshDown + 3;
%  Indown = rand (1,lenseq)*(threshUp - threshDown) + threshDown;
%  
%  Indown = [Indown , Indown(end):-sp:down];

%  fino = threshDown - 3 - abs(randn)*3;
%  Indown = [Indown , Indown(end):-sp:fino, fino:sp:10];

%  Indown = 10;
%  In = [Inup Indown];
In = Inup;
out_fe = y0;
for i=1:length(In)
	y0 = out_fe(end);
	fix = ctrnn_fe('sigmoid',y0, W, We, In(i), theta, tau, steps, deltaT);
	out_fe = [out_fe fix(end)];
end
out = out_fe;
%  figure
%  plot (sigmoid([y0,out_fe]),'.r')
%  ylim([0,1]);
end