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
tau	= 1;
iter	= 500;
step	= 0.125;
%  step	= 0.005;
In	= In_start:step:In_end;
cicli	= max(size(In));
out_n	= zeros(1,cicli);
out_p	= zeros(1,cicli);
tic
for i=1:cicli
	execute = ctrnn_fe (outFun, y0,W,We,In(i),theta,tau,iter,deltaT);
	out_p(1,i)=execute(iter);
	execute = ctrnn_fe (outFun,-y0,W,We,In(i),theta,tau,iter,deltaT);
	out_n(1,i)=execute(iter);
end
toc
for i=1:cicli
	execute = ctrnn_dual_fe (outFun, sigmoid(y0),W,We,In(i),theta,tau,iter,deltaT);
	out_p(1,i)=sigmoidInverse(execute(iter));
	execute = ctrnn_dual_fe (outFun,sigmoid(-y0),W,We,In(i),theta,tau,iter,deltaT);
	out_n(1,i)=sigmoidInverse(execute(iter));
end

figure;
hold on;box on; grid on;
plot (In,out_n,'.r');
plot (In,out_p,'.b');
xlabel('I','fontsize',15);
ylabel('$\bar{y}$','interpreter','latex','fontsize',15);
%figure;
%hold on;
%plot (In,sigmoid(out_n));
%plot (In,sigmoid(out_p));