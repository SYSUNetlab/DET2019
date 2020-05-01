function [neighbor_state_block]=AuxFun_NeighborStateSet(real_label_block, adj_node)
%��¼ÿһ��λ�ã�n,t�����ھӽڵ��״̬��ÿһ��λ��Ϊһ������������¼�ھӽڵ�״̬
[node_num, timeslot]=size(real_label_block);
neighbor_state_block=cell(node_num, timeslot);
for node_id=1:node_num
    node_ids=adj_node{node_id};
    temp=real_label_block(node_ids,:);
    %��һ��ʱ���û��ʱ���ھ�
    neighbor_state_block{node_id,1}=temp(:,1);  %������
    %�ڶ���ʱ��㿪ʼ��Ҫ���ӿռ���ھ�
    for time_id=2:timeslot
        neighbor_state_block{node_id,time_id}=[temp(:,time_id);real_label_block(node_id,time_id-1)];   % ע�⣬����ֻ��ȡ�˿ռ��ϵ��ھӶ��ѣ�����Ҫ����ʱ���ھ�״̬�� ��������
    end
end

end