function [estimate_state_block]=AuxFun_EstimateStateBlock(estimate_state_series_block,node_num)
%�ҳ����Ƴ�����״̬��
[threefold_samples_num,folds]=size(estimate_state_series_block);
fold_samples_num=threefold_samples_num/3;
timeslot_perfold=fold_samples_num/node_num;
estimate_state_block=estimate_state_series_block(1:fold_samples_num,:);
estimate_state_block=reshape(estimate_state_block,[node_num,timeslot_perfold*folds]);

end