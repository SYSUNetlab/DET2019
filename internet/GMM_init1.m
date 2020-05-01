function [mu1,sigma1,mu2,sigma2,S,GMMmodel]=GMM_init1(train_data1,train_data2,state_num)
% 训练数据中第一个观测值是目的IP信息熵，第二个观测值是到达率。
[node_num,timeslot_num]=size(train_data1);
train1=train_data1(:);
train2=train_data2(:);
train_data=[train1,train2];
GMMmodel=fitgmdist(train_data,state_num);
P=posterior(GMMmodel,train_data);
[~,label]=max(P,[],2);
temp_mu=zeros(state_num,1);
for i=1:state_num
    temp_mu(i)=mean(train2(label==i));
end
temp=[temp_mu,[1:state_num]'];
temp=sortrows(temp,1); % 根据第一列排序
temp_state=label;
for i=1:state_num
    temp_state(label==temp(i,2))=i;
end
S=reshape(temp_state,[node_num,timeslot_num]);

for i=1:state_num
    oo=train1(temp_state==i);
    mu1(i)=mean(oo);
    sigma1(i)=std(oo);
end
for i=1:state_num
    oo=train2(temp_state==i);
    mu2(i)=mean(oo);
    sigma2(i)=std(oo);
end

end