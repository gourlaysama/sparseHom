% Copyright (c) 2012, Antoine Gourlay
% All rights reserved.
%
% This file is distributed under the Simplified BSD License;
% see the provided LICENSE file.

function [ u, lambda ] = sparsehom( y, H, K, biaised, display )
%SPARSEHOM Homotopy continuation-based method for sparse signal
%  representation in overcomplete dictionaries.
%
%   [ uout, lambda ] = sparsehom( y, H, K )
%   [ uout, lambda ] = sparsehom( y, H, K, biaised )
%   [ uout, lambda ] = sparsehom( y, H, K, biaised, display )
%
% Solves a system y = A*u + b by minimizing ||y - A*u|| + h*norm(u,1)
% with u sparse.
%
% Inputs:
% y: data vector
% H: dictionary matrix
% K: number of non-zero componentes allowed in u
% biaised: 0 for unbiaised solution (default)
%          1 for biaised solution
% display: 0 = no display (default value)
%          1 = print various details to the console
%          2 = full graphic display (slower) + console details
%
% Outputs:
% uout: minimum representation
% lambda: last value of lambda

di = false; pl = false; biais = false;

if nargin == 5
    di = display >= 1;
    pl = display == 2;
    biais = biaised > 0;
elseif nargin == 4
    biais = biaised > 0;    
end

[m, n] = size(H);

% initialisation
step = 1;
thy = H'*y;
hb = abs(thy);
[lambda, i] = max(hb); % indice of the first element to become non-nul
nz = i;
u = zeros(n,1); u(i) = thy(i);
z = 1:n; z(i) = [];
k = n-1;
s = sign(thy(i));

seye = speye(m);

clear thy hb;
if pl
    allu = zeros(n,1);
    alllambda = lambda;
    err = norm(y)^2;
end

if di
    disp(['First lambda: ', num2str(lambda)]);
end

if K == 0
    u = zeros(n,1);
    if di
        disp('Finished in 0 steps.');
    end
else
    if di
        disp(['A component of u became non-zero. Index: ',num2str(i)]);
    end
    cont = true;
    while cont
        % 1st case: u has a new non-zero component
        Hnz = H(:,nz);
        a = pinv(Hnz);
        yt = (seye-Hnz*a)*y;
        d = a'*s;
        
        Hz = H(:,z);
        num = Hz'*yt;
        den0 = Hz'*d;
        v = zeros(k,2);
        for j = 1:k
            v(j,1) = num(j)/(1-den0(j));
            v(j,2) = -num(j)/(1+den0(j));
        end
        v = v(:);
        [v, mi] = sort(v,'descend');
        idx = find(v<lambda-1e-10,1,'first');
        la1 = v(idx);
        idx = mi(idx);
        if idx > k
            i1 = idx-k;
            ep = -1;
        else
            i1= idx;
            ep = 1;
        end
        if la1 > lambda - 1e-10
            la1 = 0;
        end
        
        % 2nd case: u has a new zero component
        num2 = a*y;
        den = Hnz'*Hnz;
        den = den\s;
        w = num2./den;
        [w, mi] = sort(w,'descend');
        i2 = find(w<lambda-1e-10,1,'first');
        la2 = w(i2);
        i2 = mi(i2);
        if la2 > lambda - 1e-10
            la2 = 0;
        end
        
        % pick the right case
        [nlambda, cas] = max([la1 la2]);
        
        if abs(nlambda) < 1e-10
            lambda = 0;
            
            if di
                disp(['Reached lambda = 0 with ',num2str(n-k),' non-zeros components in u,']);
                disp(['after ',num2str(step),' iterations. This is the last sparse solution. Exiting.']);
            end
            u(nz) = num2;
            u(z) = zeros(k,1);
            if pl
                allu = [allu u];
                err = [err; norm(y - H*u)^2];
                alllambda = [alllambda, 0];
            end
            break;
        end
        
        lambda = nlambda;
        if di
            disp(['New lambda: ',num2str(nlambda)]);
        end
        
        if pl
            u(z) = zeros(k,1);
            u(nz) = num2; % unbiaised u
            
            err = [err; norm(y - H*u)^2];
            alllambda = [alllambda, lambda];
            
            u(nz) = u(nz) - (lambda(end))*den; % biaised u
            allu = [allu u];
        end
        
        cont = n-k+1 <= K;
        if cont
            % update variables
            step = step + 1;
            if cas == 1
                if di
                    disp(['A component of u became non-zero. Index: ',num2str(z(i1))]);
                end
                temp = z(i1);
                z(i1) = [];
                nz = [nz, temp];
                s = [s; ep];
                k = k-1;
            else
                if di
                    disp(['A component of u became zero again. Index: ',num2str(nz(i2))]);
                end
                temp = nz(i2);
                nz(i2) = [];
                z = [z, temp];
                s(i2) = [];
                k = k+1;
            end
            
        else
            if biais
                u(z) = zeros(k,1);
                u(nz) = num2 - lambda*den;
            else
                u(z) = zeros(k,1);
                u(nz) = num2;
            end
            if di
                disp(['Reached  ',num2str(K),' non-zero components in u,']);
                disp(['after  ',num2str(step),' iterations. Exiting.']);
            end
        end
    end
end

if pl && K > 0
    figure;
    subplot(211);
    alllambda = log10(alllambda);
    
    tt = abs(allu');
    [~,j] = find(sum(tt));
    tt = tt(:,j);
    m = max(tt(:));
    na = size(tt,2);
    xx = repmat(alllambda',1,na);
    if lambda == 0
        start = alllambda(end-1);
    else
        start = alllambda(end);
    end
    axis([start alllambda(1) 0 m*1.1]);
    line(xx,tt,'Marker', 'd', 'LineWidth', 2);
    xx = repmat(alllambda,2,1);
    tt = repmat([0; m*1.1],1,length(alllambda));
    line(xx,tt,'LineStyle','--', 'Color', 'k');
    xt = linspace(start,alllambda(1),10);
    set(gca,'XTick',xt);
    xtl = cellstr(num2str(10.^xt(:),'%.1f'));
    set(gca,'XTickLabel',xtl);
    xlabel('Lambda values (log scale)');
    ylabel('abs(u_i) for non-zero components');
    title('Evolution of non-zero components with lambda');
    
    subplot(212);
    h = plot(alllambda',err,'d-', 'LineWidth', 2);
    m = max(err);
    tt = repmat([0; m*1.1],1,length(alllambda));
    line(xx,tt,'LineStyle','--', 'Color', 'k');
    axis([start alllambda(1) 0 m*1.1]);
    set(gca,'XTick',xt);
    set(gca,'XTickLabel',xtl);
    xlabel('Lambda values (log scale)');
    ylabel('norm(y_{ortho})^2');
    title('Evolution of norm(y_{ortho})^2 with lambda');
end

end

