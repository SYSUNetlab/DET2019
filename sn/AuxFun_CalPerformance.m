function [ confusion_matrix, performance_index ] = AuxFun_CalPerformance( real_state, estimate_state, state_num )
%   将真实状态序列real_state(列向量)与估计状态序列estimate_state(列向量)作对比，比较估计的性能
%   性能指标主要包括准确率，错误率，精准率，召回率，F1值，混淆矩阵
%   想法：是否可以定义一种评判标准，将其状态估计错误偏离程度给评定出来。
sample_num=length(real_state);
difference=real_state-estimate_state;
error_num=length(find(difference~=0));
error_rate=error_num/sample_num;
accurate_rate=1-error_rate;
%针对于不同的状态计算Precision，Recall，F1
confusion_matrix=zeros(state_num);
for i=1:state_num
    temp_estimate_result=estimate_state(real_state==i);
    for j=1:state_num
        estimate_id=find(temp_estimate_result==j);
        temp_estimate_num=length(estimate_id);
        confusion_matrix(i,j)=temp_estimate_num;
    end
end
%性能指标，第一行为总的准确率，第二行为总的错误率，第三行为每一类的精确率precision，第四行为每一类的召回率recall，第五行为每一类的F1值，最后一行加上FPR
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

