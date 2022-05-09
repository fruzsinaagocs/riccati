% check integral of |a-exp(i.theta)|^-2 over theta in [0,2pi), where a = 1-1/k
% Barnett 5/9/22
clear
k = 17;  % to test

n=ceil(30*k);  % convergence
I = pi*k/(1-0.5/k)
t=(1:n)/n*2*pi; h = 2*pi/n;  % PTR
a = 1-1/k;
f = @(t) abs(a-exp(1i*t)).^-2;
In = h*sum(f(t))
I-In
