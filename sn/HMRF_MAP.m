%The MAP algorithm
%--------input-------------------
% S:inital state
% mu:vector of means
% k:number of labels
% MAP_iter: maximum number of iterations of MAP algorithm
% C:connection of nodes
% alpha: model spatial parameter
% beta:model temporal parameter
%--------output------------------
% S:final labeled state
% sum_U:final sum energy
function [S,sum_U_MAP]=HMRF_MAP(S,O,mu,sigma,k,MAP_iter,C,alpha,beta,show_plot)
sum_U_MAP=zeros(1,MAP_iter);
[p q]=size(S);
for it=1:MAP_iter % iterations
    fprintf('  Inner iteration: %d\n',it);
    [m n]=size(S);
    o=O(:);
    U=zeros(m*n,k);
    U1=U;
    U2=U;
    for l=1:k % all labels         
        temp1=normpdf(o,mu(l),sigma(l));
        U1(:,l)=U1(:,l)+temp1;
        for ind=1:m*n
% the first type of clique potential function
            [i j]=ind2ij(ind,m);
            u2=0;
            %compute the spatial energy
            neighbor=C{i};
            len=length(neighbor);
            temp2=0;
            for nei=1:len
                if l~=S(neighbor(nei),j)
                    temp2=temp2+alpha;
                end
            end
            temp2=temp2/len;
            %compute the temporal energy
            temp3=0;
            if j-1>=1 && l~=S(i,j-1)
                temp3=temp3+beta;
            end
            u2=exp(-temp2-temp3);
            U2(ind,l)=u2;

    % the first type of clique potential function
%                 [i j]=ind2ij(ind,m);
%                 u2=0;
%                 neighbor=C{i};
%                 len=length(neighbor);
%                 temp2=0;
%                 for nei=1:len
%                     temp2=temp2+alpha*abs(S(neighbor(nei),j)-l);
%                 end
%                 temp2=temp2/len;
%                 temp3=0;
%                 if j-1>=1
%                     temp3=temp3+beta*abs(S(i,j-1)-l);
%                 end
%                 u2=exp(-temp2-temp3);
%                 U2(ind,l)=u2;
        end   
    end
    U1=bsxfun(@rdivide,U1,sum(U1,2));
    %compute the margin Z function
    U2=bsxfun(@rdivide,U2,sum(U2,2));
    %compute the sum posteriori probability
    U=bsxfun(@times,U1,U2);
    %optimize the infer hidden state
    [temp s]=max(U,[],2);
    temp=log(temp);
    sum_U_MAP(it)=sum(temp(:));
    S=reshape(s,[p q]);
    if it>=10 && std(sum_U_MAP(it-2:it))/abs(sum_U_MAP(it))<0.001
        break;
    end
    
end
sum_U=0;
for ind=1:p*q
    sum_U=sum_U+(U(ind,s(ind)));
end

if show_plot==1
    figure;
    hold on;
    plot(1:it,sum_U_MAP(1:it),'r');
    title('sum posteriori probability MAP');
    xlabel('MAP iteration');
    ylabel('sum posteriori probability MAP');
    drawnow;
end
sum_U_MAP(find(sum_U_MAP==0))=[];
end


function [i j]=ind2ij(ind,m)
i=mod(ind-1,m)+1;
j=floor((ind-1)/m)+1;
end