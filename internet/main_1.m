%% germany50 ddos1
%
clear variables;
state_num =3;
load('.\data\germany50_topo.mat');
load('.\data\germany50_sourcedata_ddos1.mat');
[germany50_performance_struct_ddos1]=AuxFun_CrossValidataion4( germany50_dst_entropy_ddos1, germany50_arrival_rate_ddos1, germany50_data_label_ddos1_block, germany50_adjacent_nodes_list, state_num);
save('.\data\germany50_performance_struct_ddos1V1.mat','germany50_performance_struct_ddos1');
temp_data=germany50_performance_struct_ddos1.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3);

load('.\data\germany50_performance_struct_ddos1V1.mat')
performance_index_em=germany50_performance_struct_ddos1.performance_index_em;
performance_em_pertime=germany50_performance_struct_ddos1.performance_em_pertime;
confusion_matrix_em_pertime=germany50_performance_struct_ddos1.confusion_matrix_em_pertime;
em_estimate_state_series_block=germany50_performance_struct_ddos1.em_estimate_state_series_block;
parameter_em_pertime=germany50_performance_struct_ddos1.parameter_em_pertime;
likelihood_function_pertime=germany50_performance_struct_ddos1.likelihood_function_pertime;
save('.\data2\germany50_performance_struct_ddos1_pyV1.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');

%}

%% germany50 ddos2
%
clear variables;
state_num =3;
load('.\data\germany50_topo.mat');
load('.\data\germany50_sourcedata_ddos2.mat');
[germany50_performance_struct_ddos2]=AuxFun_CrossValidataion4( germany50_dst_entropy_ddos2, germany50_arrival_rate_ddos2, germany50_data_label_ddos2_block, germany50_adjacent_nodes_list, state_num);
save('.\data\germany50_performance_struct_ddos2V1.mat','germany50_performance_struct_ddos2');
temp_data=germany50_performance_struct_ddos2.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)

load('.\data\germany50_performance_struct_ddos2V1.mat')
performance_index_em=germany50_performance_struct_ddos2.performance_index_em;
performance_em_pertime=germany50_performance_struct_ddos2.performance_em_pertime;
confusion_matrix_em_pertime=germany50_performance_struct_ddos2.confusion_matrix_em_pertime;
em_estimate_state_series_block=germany50_performance_struct_ddos2.em_estimate_state_series_block;
parameter_em_pertime=germany50_performance_struct_ddos2.parameter_em_pertime;
likelihood_function_pertime=germany50_performance_struct_ddos2.likelihood_function_pertime;
save('.\data2\germany50_performance_struct_ddos2_pyV1.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');

%}

%% scalefree64 ddos1
%
clear variables;
state_num =3;
load('.\data\scalefree64_topo.mat');
load('.\data\scalefree64_sourcedata_ddos1_newV1.mat');
[scalefree64_performance_struct_ddos1]=AuxFun_CrossValidataion4( scalefree64_dst_entropy_ddos1, scalefree64_arrival_rate_ddos1, scalefree64_data_label_ddos1_block, scalefree64_adjacent_nodes_list, state_num);
save('.\data\scalefree64_performance_struct_ddos1_newV1.mat','scalefree64_performance_struct_ddos1');
temp_data=scalefree64_performance_struct_ddos1.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)
%}
%% scalefree64 ddos1
%
%load('.\data\scalefree64_performance_struct_ddos1V1.mat')
performance_index_em=scalefree64_performance_struct_ddos1.performance_index_em;
performance_em_pertime=scalefree64_performance_struct_ddos1.performance_em_pertime;
confusion_matrix_em_pertime=scalefree64_performance_struct_ddos1.confusion_matrix_em_pertime;
em_estimate_state_series_block=scalefree64_performance_struct_ddos1.em_estimate_state_series_block;
parameter_em_pertime=scalefree64_performance_struct_ddos1.parameter_em_pertime;
likelihood_function_pertime=scalefree64_performance_struct_ddos1.likelihood_function_pertime;
save('.\data2\scalefree64_performance_struct_ddos1_py_newV1.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');
%}


%% scalefree64 ddos2
clear variables;
state_num =3;
load('.\data\scalefree64_topo.mat');
load('.\data\scalefree64_sourcedata_ddos2_V1_add.mat');
[scalefree64_performance_struct_ddos2]=AuxFun_CrossValidataion4( scalefree64_dst_entropy_ddos2, scalefree64_arrival_rate_ddos2, scalefree64_data_label_ddos2_block, scalefree64_adjacent_nodes_list, state_num);
save('.\data\scalefree64_performance_struct_ddos2V1_add.mat','scalefree64_performance_struct_ddos2');
temp_data=scalefree64_performance_struct_ddos2.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)
%}
%% scalefree64 ddos2
%
%load('.\data\scalefree64_performance_struct_ddos2V2.mat')
performance_index_em=scalefree64_performance_struct_ddos2.performance_index_em;
performance_em_pertime=scalefree64_performance_struct_ddos2.performance_em_pertime;
confusion_matrix_em_pertime=scalefree64_performance_struct_ddos2.confusion_matrix_em_pertime;
em_estimate_state_series_block=scalefree64_performance_struct_ddos2.em_estimate_state_series_block;
parameter_em_pertime=scalefree64_performance_struct_ddos2.parameter_em_pertime;
likelihood_function_pertime=scalefree64_performance_struct_ddos2.likelihood_function_pertime;
save('.\data2\scalefree64_performance_struct_ddos2_pyV1_add.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');
%}