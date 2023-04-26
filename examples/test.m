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


In = [-5:sp:up , up:-sp:-5]; % In = [0:-5:-20 , -20:5:0];

lenseq = round(200 + randn*50);

Inup = rand (1,lenseq)*(threshUp - threshDown) + threshDown;
fino = threshUp + 3 + abs(randn)*3;
Inup = [Inup , Inup(end):sp:fino, fino:-sp:10];

lenseq = round(200 + randn*50);

Indown = rand (1,lenseq)*(up - threshDown) + threshDown + 3;
Indown = rand (1,lenseq)*(threshUp - threshDown) + threshDown;

Indown = [Indown , Indown(end):-sp:down];

fino = threshDown - 3 - abs(randn)*3;
Indown = [Indown , Indown(end):-sp:fino, fino:sp:10];

%  Indown = 10;
In = [Inup Indown];

out_fe = y0;
for i=1:length(In)
	y0 = out_fe(end);
	out_fe = [out_fe ctrnn_fe('sigmoid',y0, W, We, In(i), theta, tau, steps, deltaT)];
end

figure
plot (sigmoid([y0,out_fe]),'.r')
ylim([0,1]);