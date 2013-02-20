clear variables; close all;

rng(12);
A = 2*rand(50,400)-1;
b = rand(50,1);
uorig = 2*rand(400,1)-1;
uorig(randperm(400,300)) = zeros(300,1);

d = A*uorig;
sigD = std(d);
RSB = 5;
sigB = sigD / 10^(RSB/20);

y = d + sigB*b;

[u, lambda] = sparsehom(y, A, 100, 2);