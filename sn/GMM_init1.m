function [mu,sigma,S,GMMmodel]=GMM_init1(train_data,state_num)
% 训练数据中第一个观测值是目的IP信息熵，第二个观测值是到达率。
[node_num,timeslot_num]=size(train_data);
train=train_data(:);
train_data=train;
GMMmodel=fitgmdist(train_data,state_num);
P=posterior(GMMmodel,train_data);
[~,label]=max(P,[],2);
temp_mu=zeros(state_num,1);
for i=1:state_num
    temp_mu(i)=mean(train(label==i));
end
temp=[temp_mu,[1:state_num]'];
temp=sortrows(temp,1); % 根据第一列排序
temp_state=label;
for i=1:state_num
    temp_state(label==temp(i,2))=i;
end
S=reshape(temp_state,[node_num,timeslot_num]);

for i=1:state_num
    oo=train(temp_state==i);
    mu(i)=mean(oo);
    sigma(i)=std(oo);
end

end