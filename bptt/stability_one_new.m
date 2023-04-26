clear all
outFun	='sigmoid';
y0	=20;
W	=20;
We 	= 1;
In_start=-20;
In_end	= 0;
In_start=-17;
In_end	= -3;


theta	= 0;
deltaT	= 0.2;
tau     = 1;
iter	= 500;
step	= 0.125;
%  step	= 0.005;
In	= In_start:step:In_end;
cicli	= max(size(In));
out_n	= zeros(1,cicli);
out_p	= zeros(1,cicli);
net1 = createCTRNN(1,1,theta,W,We);
% net1.indexOut = 1;
net1.Ys     = zeros(iter,net1.numUnits);
net1.dt     = deltaT;
net2=net1;
net1.initialYValues =  y0;
net2.initialYValues = -y0;
net1.Ys(1,:) = net1.initialYValues;
net2.Ys(1,:) = net2.initialYValues;

tic
for i=1:cicli
% 	fprintf('cicli%g/%g\n',i,cicli);
    net1.indexOut       = 1;
    net2.indexOut       = 1;
    net1.externalInput  = In(i);
    net2.externalInput  = In(i);
    
    for k=1:iter
        net1       = ctrnn_fe_oneStep (net1);
%         executeOld = ctrnn_fe (outFun, y0,W,We,In(i),theta,tau,1,deltaT);
%         executeOld(1)
%         net1.Ys(k+1)
%         pause
        net2       = ctrnn_fe_oneStep (net2);
%         y0 =  executeOld(1);
    end
    
    execute     = net1.Ys;
%     execute(iter+1)=executeOld;
	out_p(1,i)  = execute(iter+1);
    
    execute     = net2.Ys;
	out_n(1,i)  = execute(iter+1);
    
end
toc
figure;
hold on;box on; grid on;
plot (In,out_n,'.r');
plot (In,out_p,'.b');
xlabel('I','fontsize',15);
ylabel('$\bar{y}$','interpreter','latex','fontsize',15);
hold off
%figure;
%hold on;
%plot (In,sigmoid(out_n));
%plot (In,sigmoid(out_p));