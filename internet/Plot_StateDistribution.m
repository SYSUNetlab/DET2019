clear variables;
state_num=3;
%%
load('.\data\germany50_sourcedata_ddos1.mat');
data_cell=cell(state_num,2);
for i=1:state_num
    idx=find(germany50_data_label_ddos1_block==i);
    data_cell{i,1}=germany50_dst_entropy_ddos1(idx);
	data_cell{i,2}=germany50_arrival_rate_ddos1(idx);
end
germany50_ddos1_state1_dst_entropy=data_cell{1,1};
germany50_ddos1_state1_arrival_rate=data_cell{1,2};
germany50_ddos1_state2_dst_entropy=data_cell{2,1};
germany50_ddos1_state2_arrival_rate=data_cell{2,2};
germany50_ddos1_state3_dst_entropy=data_cell{3,1};
germany50_ddos1_state3_arrival_rate=data_cell{3,2};
save('.\state_distribution_data\germany50_ddos1V2.mat','germany50_ddos1_state1_dst_entropy','germany50_ddos1_state1_arrival_rate','germany50_ddos1_state2_dst_entropy','germany50_ddos1_state2_arrival_rate','germany50_ddos1_state3_dst_entropy','germany50_ddos1_state3_arrival_rate');