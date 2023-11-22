alpha = 2e-5;%linspace(1e-6, 2e-5, 10000); %positive
k = 0.5;%linspace(0.1, 10, 10000);
m = linspace(0,0.999999,10000);
L = 0.9;%linspace (0.1, 100, 10000);
Pm = (0.5e-5)^2*20;%linspace(0,42,10000);
I0 = 300;%linspace(100,1000,10000);
P = zeros(size(m));
for i=1:length(m)
    a = alpha*k/(1-m(i))*I0;
    num = a + Pm;
    den = a*exp(-1*k*L)+Pm;
    P(i) = (Pm/k)*log(num/den);
end

um = 0.5;
wg = 0.05;%linspace(0.01, 1, 10001);
ws = linspace(0.01,1,10001);%0.03;
u = zeros(size(ws));

for i=1:length(ws)
    w = ws(i)+wg;
    u(i) = um*ws(i)/w;
end

plot(ws,u)