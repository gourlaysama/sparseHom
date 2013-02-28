clear variables; close all;

rng(12);
A = 2*rand(5,15)-1;
b = rand(5,1);
uorig = 2*rand(15,1)-1;
uorig(randperm(15,10)) = zeros(10,1);

d = A*uorig;
sigD = std(d);
RSB = 15;
sigB = sigD / 10^(RSB/20);

y = d + sigB*b;

[u, lambda] = sparsehom(y, A, 5, false, 2);