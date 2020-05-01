function [performance_struct ]=AuxFun_CrossValidataion( observe_data1, observe_data2, label_block, adj_node, state_num )
%   ����������auxiliary function ������֤������ʮ�ε�ʮ�۽�����֤��
%   �������ݣ�observe_data ��ÿ���ڵ�Ĺ۲����ݣ�������֤ʱ���۲����ݷֳ�10�ݣ�����7������ѵ����3�����ڲ��ԡ�

original_observe_data1=observe_data1;  % ��¼ԭʼ������
original_observe_data2=observe_data2;
original_label_block=label_block;
start_time1=clock;
folds=10;  %
test_fold_num=1;
repeat_times =1;
[node_num,timeslot]=size(observe_data1);
timeslot_perfold=timeslot/folds;
%observe_data1=repmat(observe_data1,1,2);   %�ظ����ݣ��������� ������dst_entropy
%observe_data2=repmat(observe_data2,1,2);   %�ظ����ݣ���������  ������arrival_rate
index_set=1:size(observe_data2,2);

performance_em_pertime=zeros(5,state_num,repeat_times*folds);
confusion_matrix_em_pertime=zeros(state_num,state_num,repeat_times*folds);
em_estimate_state_series_block=zeros(node_num*test_fold_num*timeslot_perfold,repeat_times*folds); %���ڴ洢em�㷨���Ƶõ���״̬���С�ÿһ��Ϊ3��fold�Ĺ���״̬��

for repeat_id=1:repeat_times
    % ��ԭʼ���ݿ���shuffle�����ǲ�Ҫ�ı�ʱ���ϵ������ԡ�
    time_shift=randi([0,timeslot]);
    observe_data1 = circshift(original_observe_data1,[0,time_shift]);
    observe_data2 = circshift(original_observe_data2,[0,time_shift]);
    label_block = circshift(original_label_block,[0,time_shift]);
    %shuffle֮�󣬶����ݿ���и��ƣ�������������
    observe_data1=repmat(observe_data1,1,2);   %�ظ����ݣ��������� ������dst_entropy
    observe_data2=repmat(observe_data2,1,2);   %�ظ����ݣ���������  ������arrival_rate
    label_block=repmat(label_block,1,2);  %�ظ����ݣ���������
    
    for fold_id=1:folds
        test_index=(fold_id-1)*timeslot_perfold+1:(fold_id+(test_fold_num-1))*timeslot_perfold;    % test_fold_num��fold��Ϊ��������
        temp_index_set=(fold_id-1)*timeslot_perfold+1:(fold_id+9)*timeslot_perfold;   %indexÿ�������ƶ�һ��
        train_index=setdiff(temp_index_set,test_index);
        training_data1=observe_data1(:,train_index);
        training_data2=observe_data2(:,train_index);
        test_data1=observe_data1(:,test_index);
        test_data2=observe_data2(:,test_index);
        test_label_block=label_block(:,test_index);
        [p,q]=size(test_data2);

        %%ģ��ѵ��
        tic;
        %����˵�� [ state_block_em, parameter_em ,likelihood_function] = Model_HMRF_ParameterEstimation( training_data1, training_data2, adj_node, state_num )
        [ ~, parameter_em, likelihood_function ] = Model_HMRF_ParameterEstimation( training_data1, training_data2, adj_node, state_num );
        toc;
        idxequals1=find(parameter_em(1,:)==1)
        if isempty(idxequals1)
            model_parameters=parameter_em(:,end);
        else
            model_parameters=parameter_em(:,idxequals1(1)-1);
        end
        % em �ƶϷ���
        % �ƶ�״̬������������Ҫ����ת������������ͬʱҪ��ʼ��״̬��,�����ʼ��״̬������Ҫʹ��kmeans�ˣ�ֱ�Ӱ���ѵ���õ���״̬�ֲ�ȥ��ʼ��
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
        %����˵��[ update_state_series ] = Model_ICM_Infer_SF( observation_series, state_series, parameters, adj_node )

        %%kmeans ����
        % [ kmeans_estimate_state_series, ~] = AuxFun_KmeansClustering( test_data1(:), state_num);

        %******************************************************************************
        %����״̬��ǩ
        test_real_state_series_allnode=test_label_block(:);

        %��ͬ������������
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


        %���ÿһ����֤�Ľ��
        fprintf('performance_em_pertime in %s th:\n',num2str((repeat_id-1) * repeat_times + fold_id));
        performance_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)
        %performance_kmeans_pertime(:,:,fold_id);
        fprintf('confusion_matrix_em_pertime in %s th:\n',num2str((repeat_id-1) * repeat_times + fold_id));
        confusion_matrix_em_pertime(:,:,(repeat_id-1) * repeat_times + fold_id)
        %confusion_matrix_kmeans_pertime(:,:,fold_id);
        toc;

    end

end
%���ѵ����֤����ƽ����cross validation
performance_index_em=mean(performance_em_pertime,3);   %�ڵ�����ά����ȡ��ֵ
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

