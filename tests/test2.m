clear all; close all;

rng(12);
A = rand(5,15);
b = rand(5,1);
uorig = rand(15,1);
uorig(randperm(15,10)) = zeros(10,1);

d = A*uorig;
sigD = std(d);
RSB = 15;
sigB = sigD / 10^(RSB/20);

y = d + sigB*b;

[u, lambda] = sparsehom(y, A, 10, 2);