function [S,mu1,sigma1,mu2,sigma2,alpha,beta,sum_U] = HMRF_EM_simultaneously(S,O1,O2,mu1,sigma1,mu2,sigma2,k,C,EM_iter,MAP_iter,alpha,beta)
[m, n]=size(O1);
o1=O1(:);
o2=O2(:);
sum_U=zeros(1,EM_iter);
sum_EM=zeros(1,EM_iter);
for it=1:EM_iter
    %fprintf('Iteration: %d\n',it);
    %% update S
    [S,sum_U_MAP]=HMRF_MAP(S,O1,O2,mu1,sigma1,mu2,sigma2,k,MAP_iter,C,alpha,beta,0);
    sum_U(it)=sum_U_MAP(end);
    s=S(:);
%     a=sym('a');
%     b=sym('b');
    %% compute Q
    U=zeros(m*n,k);
    U_current=U;
    U_o=U;
%     U_next=zeros(m*n,k,'sym');
    for l=1:k
        temp0=normpdf(o1,mu1(l),sigma1(l))+eps;
        temp1=normpdf(o2,mu2(l),sigma2(l))+eps;
        U_o(:,l)=U_o(:,l)+temp0.*temp1;
        for ind=1:m*n % all labels
            %compute u=a_{t,n} and temp1=b_m based on parameter obtained in the
            %i_th iteration
            [i j]=ind2ij(ind,m);
            u=0;
            %compute the spatial energy
            neighbor=C{i};
            len=length(neighbor);
            temp2=0;
%             sym_temp2=sym('0');
            for nei=1:len
%                 temp2=temp2+alpha*(S(neighbor(nei),j)-l)^2;
                %sym_temp2=sym_temp2+a*(S(neighbor(nei),j)-l)^2;
                if l~=S(neighbor(nei),j)
                    temp2=temp2+alpha;
%                     sym_temp2=sym_temp2+a;
                end
            end
            temp2=temp2/len;
%             sym_temp2=sym_temp2/len;
            %compute the temporal energy
            temp3=0;
%             sym_temp3=sym('0');
            if j-1>=1 && l~=S(i,j-1)
                temp3=temp3+beta;
%                 sym_temp3=sym_temp3+b;
            end
%             if j-1>=1
%                 temp3=temp3+beta*(S(i,j-1)-l)^2;
                %sym_temp3=sym_temp3+b*(S(i,j-1)-l)^2;
%             end
            u=exp(-temp2-temp3);
            %u_next=exp(-sym_temp2-sym_temp3);
            U_current(ind,l)=u;
            %U_next(ind,l)=u_next;
        end
    end
    U_current=bsxfun(@rdivide,U_current,sum(U_current,2));
    U_o=bsxfun(@rdivide,U_o,sum(U_o,2));
    Q_o=bsxfun(@times,U_current,U_o);
%     lkh = log(Q_o);
%     sum_EM(it)=sum(lkh(:));
%     U_next=U_next./repmat(sum(U_next,2),1,k);
%     U_next=log(U_next);
%     Q_A=Q_o.*U_next;
%     Q_A=sum(Q_A(:));
%     normalize=zeros(m*n,1,'sym');
%     for q=1:k
%         q
%         parfor p=1:m*n
%             normalize(p)=normalize(p)+U_next(p,q);
%         end
%     end
%     Q_A=sym('0');
%     for q=1:k
%         q
%         parfor p=1:m*n
%             U_next(p,q)=U_next(p,q)/normalize(p);
%             Q_A=Q_A+Q_o(p,q)*log(U_next(p,q));
%         end
%     end
    %[parameter,fval]=max_function(Q_A,[alpha,beta])
    %sum_EM(it)=fval;
    if it>=6 && std(sum_U(it-3:it))/abs(sum_U(it))<0.0001
        break;
    end
    %alpha=parameter(1);
    %beta=parameter(2);
    for l=1:k
        mu1(l)=o1'*Q_o(:,l)/sum(Q_o(:,l));
        sigma1(l)=((o1-mu1(l)).^2)'*Q_o(:,l)/sum(Q_o(:,l));
        sigma1(l)=sqrt(sigma1(l));
        
        mu2(l)=o2'*Q_o(:,l)/sum(Q_o(:,l));
        sigma2(l)=((o2-mu2(l)).^2)'*Q_o(:,l)/sum(Q_o(:,l));
        sigma2(l)=sqrt(sigma2(l));
    end
end

% figure;
% plot(1:it,sum_U(1:it),'r');
% title('EM iteration');
% xlabel('EM iteration');
% ylabel('sum LKH');
% drawnow;
end

function [i j]=ind2ij(ind,m)
i=mod(ind-1,m)+1;
j=floor((ind-1)/m)+1;
end