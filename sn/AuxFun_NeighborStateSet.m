function [neighbor_state_block]=AuxFun_NeighborStateSet(real_label_block, adj_node)
%记录每一个位置（n,t）的邻居节点的状态，每一个位置为一个列向量，记录邻居节点状态
[node_num, timeslot]=size(real_label_block);
neighbor_state_block=cell(node_num, timeslot);
for node_id=1:node_num
    node_ids=adj_node{node_id};
    temp=real_label_block(node_ids,:);
    %第一个时间点没有时间邻居
    neighbor_state_block{node_id,1}=temp(:,1);  %列向量
    %第二个时间点开始需要增加空间的邻居
    for time_id=2:timeslot
        neighbor_state_block{node_id,time_id}=[temp(:,time_id);real_label_block(node_id,time_id-1)];   % 注意，这里只是取了空间上的邻居而已，还需要增加时间邻居状态。 现已增加
    end
end

end