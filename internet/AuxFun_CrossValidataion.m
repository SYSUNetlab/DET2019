function [performance_struct ]=AuxFun_CrossValidataion( observe_data1, observe_data2, label_block, adj_node, state_num )
%   辅助函数：auxiliary function 交叉验证，进行十次的十折交叉验证。
%   输入数据：observe_data 是每个节点的观测数据，交叉验证时，观测数据分成10份，其中7份用于训练，3份用于测试。

original_observe_data1=observe_data1;  % 记录原始的数据
original_observe_data2=observe_data2;
original_label_block=label_block;
start_time1=clock;
folds=10;  %
test_fold_num=1;
repeat_times =1;
[node_num,timeslot]=size(observe_data1);
timeslot_perfold=timeslot/folds;
%observe_data1=repmat(observe_data1,1,2);   %重复数据，方便索引 这里是dst_entropy
%observe_data2=repmat(observe_data2,1,2);   %重复数据，方便索引  这里是arrival_rate
index_set=1:size(observe_data2,2);

performance_em_pertime=zeros(5,state_num,repeat_times*folds);
confusion_matrix_em_pertime=zeros(state_num,state_num,repeat_times*folds);
em_estimate_state_series_block=zeros(node_num*test_fold_num*timeslot_perfold,repeat_times*folds); %用于存储em算法估计得到的状态序列。每一列为3个fold的估计状态。

for repeat_id=1:repeat_times
    % 对原始数据块做shuffle，但是不要改变时间上的连续性。
    time_shift=randi([0,timeslot]);
    observe_data1 = circshift(original_observe_data1,[0,time_shift]);
    observe_data2 = circshift(original_observe_data2,[0,time_shift]);
    label_block = circshift(original_label_block,[0,time_shift]);
    %shuffle之后，对数据块进行复制，方便做索引。
    observe_data1=repmat(observe_data1,1,2);   %重复数据，方便索引 这里是dst_entropy
    observe_data2=repmat(observe_data2,1,2);   %重复数据，方便索引  这里是arrival_rate
    label_block=repmat(label_block,1,2);  %重复数据，方便索引
    
    for fold_id=1:folds
        test_index=(fold_id-1)*timeslot_perfold+1:(fold_id+(test_fold_num-1))*timeslot_perfold;    % test_fold_num个fold作为测试数据
        temp_index_set=(fold_id-1)*timeslot_perfold+1:(fold_id+9)*timeslot_perfold;   %index每次向右移动一折
        train_index=setdiff(temp_index_set,test_index);
        training_data1=observe_data1(:,train_index);
        training_data2=observe_data2(:,train_index);
        test_data1=observe_data1(:,test_index);
        test_data2=observe_data2(:,test_index);
        test_label_block=label_block(:,test_index);
        [p,q]=size(test_data2);

        %%模型训练
        tic;
        %函数说明 [ state_block_em, parameter_em ,likelihood_function] = Model_HMRF_ParameterEstimation( training_data1, training_data2, adj_node, state_num )
        [ ~, parameter_em, likelihood_function ] = Model_HMRF_ParameterEstimation( training_data1, training_data2, adj_node, state_num );
        toc;
        idxequals1=find(parameter_em(1,:)==1)
        if isempty(idxequals1)
            model_parameters=parameter_em(:,end);
        else
            model_parameters=parameter_em(:,idxequals1(1)-1);
        end
        % em 推断方法
        % 推断状态场，测试数据要首先转换成列向量，同时要初始化状态场,这里初始化状态场不需要使用kmeans了，直接按照训练得到的状态分布去初始化
        %[ initialize_kmeans_estimate_state_series, ~] = AuxFun_KmeansClustering( test_data(:), state_num);
        parameters=model_parameters(1:end-1,end);
        [initialize_estimate_state_series]=AuxFun_EstimateInitialize( test_data1(:), test_data2(:), parameters );
        S_test=reshape(initialize_estimate_state_series,[p,q]);
        MAP_iter=20;
        mu1=model_parameters(1:3);
        mu2=model_parameters(4:6);
        sigma1=model_parameters(7:9);
        sigma2=model_parameters(10:12);
        alpha=model_parameters(end);
        beta=alpha;
        [S,sum_U_MAP]=HMRF_MAP(S_test,test_data1,test_data2,mu1,sigma1,mu2,sigma2,state_num,MAP_iter,adj_node',alpha,beta,1);
        em_estimate_state_series=reshape(S,[p*q,1]);
        %[ em_estimate_state_series ] = Model_ICM_Infer_SF( test_data1(:), test_data2(:), initialize_estimate_state_series, model_parameters, adj_node );
        %函数说明[ update_state_series ] = Model_ICM_Infer_SF( observation_series, state_series, parameters, adj_node )

        %%kmeans 方法
        % [ kmeans_estimate_state_series, ~] = AuxFun_KmeansClustering( test_data1(:), state_num);

        %******************************************************************************
        %数据状态标签
        test_real_state_series_allnode=test_label_block(:);

        %不同方法性能评估
        %[ confusion_matrix, performance_index ] = Performance( real_state, estimate_state, state_num )
        [ confusion_matrix_em, performance_index_em ] = AuxFun_CalPerformance(test_real_state_series_allnode, em_estimate_state_series, state_num );
        %[ confusion_matrix_kmeans, performance_index_kmeans ] = AuxFun_CalPerformance(test_real_state_series_allnode, kmeans_estimate_state_series, state_num );

        performance_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)=performance_index_em;
        %performance_kmeans_pertime(:,:,fold_id)=performance_index_kmeans;
        confusion_matrix_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)=confusion_matrix_em;
        %confusion_matrix_kmeans_pertime(:,:,fold_id)=confusion_matrix_kmeans;

        em_estimate_state_series_block(:,(repeat_id-1) * repeat_times + fold_id)=em_estimate_state_series;
        %kmeans_estimate_state_series_block(:,fold_id)=kmeans_estimate_state_series;

        parameter_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)=parameter_em;
        likelihood_function_pertime((repeat_id-1) * repeat_times + fold_id,:)=likelihood_function;


        %输出每一次验证的结果
        fprintf('performance_em_pertime in %s th:\n',num2str((repeat_id-1) * repeat_times + fold_id));
        performance_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)
        %performance_kmeans_pertime(:,:,fold_id);
        fprintf('confusion_matrix_em_pertime in %s th:\n',num2str((repeat_id-1) * repeat_times + fold_id));
        confusion_matrix_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)
        %confusion_matrix_kmeans_pertime(:,:,fold_id);
        toc;

    end

end
%多次训练验证，求平均，cross validation
performance_index_em=mean(performance_em_pertime,3);   %在第三个维度上取均值
%performance_index_kmeans=mean(performance_kmeans_pertime,3);

end_time1=clock;
run_time=etime(end_time1,start_time1);
print_meg=['cross_validation runtime =',num2str(run_time),' seconds'];
disp(print_meg);

performance_struct.performance_index_em=performance_index_em;
%performance_struct.performance_index_kmeans=performance_index_kmeans;
performance_struct.performance_em_pertime=performance_em_pertime;
%performance_struct.performance_kmeans_pertime=performance_kmeans_pertime;
performance_struct.confusion_matrix_em_pertime=confusion_matrix_em_pertime;
%performance_struct.confusion_matrix_kmeans_pertime=confusion_matrix_kmeans_pertime;
performance_struct.em_estimate_state_series_block=em_estimate_state_series_block;
%performance_struct.kmeans_estimate_state_series_block=kmeans_estimate_state_series_block;
performance_struct.parameter_em_pertime=parameter_em_pertime;
performance_struct.likelihood_function_pertime=likelihood_function_pertime;


end

