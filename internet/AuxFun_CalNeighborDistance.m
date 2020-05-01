function [avg_neighbor_distance]=AuxFun_CalNeighborDistance(neighbor_state_block, specified_state)
%计算邻居节点的距离和的均值
%% 原来的计算方式
%{
[node_num, timeslot]=size(neighbor_state_block);
sum_neighbor_distance=zeros(node_num,timeslot);
neighbor_num_list=zeros(node_num,1);
for node_id=1:node_num
    neighbor_num_list(node_id)=length(neighbor_state_block{node_id,1});
    for time_id=1:timeslot
        neighbor_state_list=neighbor_state_block{node_id,time_id};
        neighbor_distance=neighbor_state_list-specified_state;
        sum_neighbor_distance(node_id,time_id)=sum(abs(neighbor_distance));
    end
end

avg_neighbor_distance=bsxfun(@rdivide,sum_neighbor_distance,neighbor_num_list);
%}

%% 现在计算方式
[node_num, timeslot]=size(neighbor_state_block);
sum_neighbor_distance=zeros(node_num,timeslot);
neighbor_num_list=zeros(node_num,timeslot);
for node_id=1:node_num
    for time_id=1:timeslot
        neighbor_num_list(node_id,time_id)=length(neighbor_state_block{node_id,time_id});
        neighbor_state_list=neighbor_state_block{node_id,time_id};
        neighbor_distance=neighbor_state_list-specified_state;
        sum_neighbor_distance(node_id,time_id)=sum(abs(neighbor_distance));
    end
end

avg_neighbor_distance=sum_neighbor_distance./neighbor_num_list;

end