clear all; close all;

A = [1 2 3; 1 3 1.5];
y = [6;6];

[u, lambda] = sparsehom(y, A, 2, false, 2);