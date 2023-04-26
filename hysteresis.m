function y=hysteresis(u,y0,alpha,m1,D1,c1)
%function y=hysteresis(u,y0,alpha,m1,D1,c1)

if nargin<6
    c1=0.7;
    if nargin <5
        D1=0.2;
        if nargin <4
            m1=3.0;
            if nargin <3
                alpha=5.0;
                if nargin <2
                    y0=-1;
                end
            end
        end
    end
end
m=alpha*m1;
D=alpha*m1*D1;

n=length(u);
y=zeros(1,n);
xsi=zeros(1,n);
y(1)=y0;
for t=1:n-1
    xsi(t)= (D1-abs(u(t)))<=0;
    diff=u(t+1)-u(t);
    y(t+1)=y(t)+(-alpha*y(t)+m*u(t)*(1-xsi(t))+D*xsi(t)*sign(u(t)))*abs(diff)+c1*(1-xsi(t))*diff;
end
end
