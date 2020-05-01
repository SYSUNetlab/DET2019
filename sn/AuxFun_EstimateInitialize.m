function [initialize_estimate_state_series]=AuxFun_EstimateInitialize( test_data, state_mean )
%   辅助函数：auxiliary function 对测试数据进行状态初始化
%   输入数据：test_data 是每个节点的观测数据，列向量，state_mean是每一个状态的均值，列向量。
%            初始化的依据是节点观测值距离哪一个状态均值近，就将其初始化为该状态。
test_data=test_data';
difference=abs(bsxfun(@minus,test_data,state_mean));
[~,initialize_estimate_state_series]=min(difference,[],1);
initialize_estimate_state_series=initialize_estimate_state_series'; %行向量装换成列向量
end