function [initialize_estimate_state_series]=AuxFun_EstimateInitialize( test_data, state_mean )
%   ����������auxiliary function �Բ������ݽ���״̬��ʼ��
%   �������ݣ�test_data ��ÿ���ڵ�Ĺ۲����ݣ���������state_mean��ÿһ��״̬�ľ�ֵ����������
%            ��ʼ���������ǽڵ�۲�ֵ������һ��״̬��ֵ�����ͽ����ʼ��Ϊ��״̬��
test_data=test_data';
difference=abs(bsxfun(@minus,test_data,state_mean));
[~,initialize_estimate_state_series]=min(difference,[],1);
initialize_estimate_state_series=initialize_estimate_state_series'; %������װ����������
end