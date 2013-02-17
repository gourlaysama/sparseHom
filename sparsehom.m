function [ u, lambda ] = sparsehom( y, H, K, display )
%SPARSEHOM Homotopy continuation-based method for sparse signal
%  representation in overcomplete dictionaries.
%
%   [ uout, lambda ] = sparsehom( y, H, K )
%   [ uout, lambda ] = sparsehom( y, H, K, display )
%
% Solves a system y = A*u + b by minimizing ||y - A*u|| + h*norm(u,1)
% with u sparse.
%
% Inputs:
% y: data vector
% H: dictionary matrix
% K: number of non-zero componentes allowed in u
% display: 0 = no display (default value)
%          1 = print various details to the console
%          2 = full graphic display (slower) + console details
%
% Outputs:
% uout: minimum representation
% lambda: array of successive values of h (in descending order)
%         until the one that generated uout

if nargin == 3
    display = 0;
end


di = display >= 1;
pl = display == 2;

[m, n] = size(H);

% initialisation
thy = H'*y;
hb = abs(thy);
[~, i] = max(hb); % indice of the first element to become non-nul
nz = i;
u = zeros(n,1); u(i) = thy(i);
z = 1:n; z(i) = [];
k = n-1;
s = sign(thy(i));
lambda = norm(hb,Inf);

if pl
    allu = zeros(n,1);
end

if di
    disp(['First lambda: ', num2str(lambda)]);
end

if K == 0
    u = zeros(n,1);
else
    if di
        disp(['A component of u became non-zero. Index: ',num2str(i)]);
    end
    cont = true;
    while cont
        % 1st case: u has a new non-zero component
        a = pinv(H(:,nz));
        yt = (eye(m)-H(:,nz)*a)*y;
        d = a'*sign(u(nz));
        
        num = H(:,z)'*yt;
        den0 = H(:,z)'*d;
        v = zeros(k,2);
        for j = 1:k
            v(j,1) = num(j)/(1-den0(j));
            v(j,2) = -num(j)/(1+den0(j));
        end
        v = v(:);
        [la1, idx] = max(v);
        if idx > k
            i1 = idx-k;
            ep = -1;
        else
            i1= idx;
            ep = 1;
        end
        if la1 > lambda(end) - 1e-10
            la1 = 0;
        end
        
        % 2nd case: u has a new zero component
        num2 = a*y;
        den = H(:,nz)'*H(:,nz);
        den = den\s;
        w = num2./den;
        [la2, i2] = max(w);
        if la2 > lambda(end) - 1e-10
            la2 = 0;
        end
        
        % pick the right case
        [nlambda, cas] = max([la1 la2]);
        
        if abs(nlambda) < 1e-10
            lambda = [lambda, 0];
            
            if di
                disp('Reached lambda = 0. This is the last sparse solution. Exiting.');
            end
            u(nz) = num2;
            u(z) = zeros(size(z));
            if pl
                allu = [allu u];
            end
            break;
        end
        
        lambda = [lambda, nlambda];
        if di
            disp(['New lambda: ',num2str(nlambda)]);
        end
        
        u(z) = zeros(size(z));
        u(nz) = num2 - (lambda(end))*den;
        if pl
            allu = [allu u];
        end
        
        cont = n-k+1 <= K;
        if cont
            % update variables
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
            if di
                disp(['Reached  ',num2str(K),' non-zero components in u. Exiting']);
            end
        end
    end
end

if pl && K > 0
    figure;
    
    tt = abs(allu');
    [~,j] = find(sum(tt));
    tt = tt(:,j);
    m = max(tt(:));
    na = size(tt,2);
    xx = repmat(lambda',1,na);
    axis([lambda(end) lambda(1) 0 m*1.1]);
    line(xx,tt,'Marker', 'd', 'LineWidth', 2);
    xx = repmat(lambda,2,1);
    tt = repmat([0; m*1.1],1,length(lambda));
    line(xx,tt,'LineStyle','--', 'Color', 'k');
    xlabel('Lambda values');
    ylabel('abs(u(i)) for non-zero components');
    title('Evolution of non-zero components with lambda')
end

end

