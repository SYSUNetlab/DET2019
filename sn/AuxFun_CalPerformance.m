function [ confusion_matrix, performance_index ] = AuxFun_CalPerformance( real_state, estimate_state, state_num )
%   ����ʵ״̬����real_state(������)�����״̬����estimate_state(������)���Աȣ��ȽϹ��Ƶ�����
%   ����ָ����Ҫ����׼ȷ�ʣ������ʣ���׼�ʣ��ٻ��ʣ�F1ֵ����������
%   �뷨���Ƿ���Զ���һ�����б�׼������״̬���ƴ���ƫ��̶ȸ�����������
sample_num=length(real_state);
difference=real_state-estimate_state;
error_num=length(find(difference~=0));
error_rate=error_num/sample_num;
accurate_rate=1-error_rate;
%����ڲ�ͬ��״̬����Precision��Recall��F1
confusion_matrix=zeros(state_num);
for i=1:state_num
    temp_estimate_result=estimate_state(real_state==i);
    for j=1:state_num
        estimate_id=find(temp_estimate_result==j);
        temp_estimate_num=length(estimate_id);
        confusion_matrix(i,j)=temp_estimate_num;
    end
end
%����ָ�꣬��һ��Ϊ�ܵ�׼ȷ�ʣ��ڶ���Ϊ�ܵĴ����ʣ�������Ϊÿһ��ľ�ȷ��precision��������Ϊÿһ����ٻ���recall��������Ϊÿһ���F1ֵ�����һ�м���FPR
performance_index=zeros(6,state_num);
performance_index(1,:)=accurate_rate;
performance_index(2,:)=error_rate;
for i=1:state_num
    true_positive=confusion_matrix(i,i);
    false_positive=sum(confusion_matrix(:,i))-true_positive;
    false_negative=sum(confusion_matrix(i,:))-true_positive;
    true_negative=sum(sum(confusion_matrix))-true_positive-false_positive-false_negative;
    precision=true_positive/(false_positive+true_positive);
    racall=true_positive/(false_negative+true_positive);
    f1_value=2*true_positive/(2*true_positive+false_positive+false_negative);
    false_positive_rate=false_positive/(false_positive+true_negative);
    performance_index(3,i)=precision;
    performance_index(4,i)=racall;
    performance_index(5,i)=f1_value;
    performance_index(6,i)=false_positive_rate;
end
end

