function [initialize_estimate_state_series]=GMM_init2(test_data1, test_data2, GMMmodel, state_num)
% 训练数据中第一个观测值是目的IP信息熵，第二个观测值是到达率。都是列向量
test_data=[test_data1,test_data2];
P2=posterior(GMMmodel,test_data);
[~,label2]=max(P2,[],2);
temp_mu=zeros(state_num,1);
for i=1:state_num
    temp_mu(i)=mean(test_data2(label2==i));
end
temp=[temp_mu,[1:state_num]'];
temp=sortrows(temp,1); % 根据第一列排序
temp_state=label2;
for i=1:state_num
    temp_state(label2==temp(i,2))=i;
end
initialize_estimate_state_series=temp_state;
end