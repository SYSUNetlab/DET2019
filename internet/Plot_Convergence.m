state_num=3;
%loading data
load('.\data\germany50_topo.mat');
load('.\data\germany50_sourcedata_ddos1.mat');
[germany50_performance_struct_ddos1]=AuxFun_CrossValidataion2_lkh( germany50_dst_entropy_ddos1, germany50_arrival_rate_ddos1, germany50_data_label_ddos1_block, germany50_adjacent_nodes_list, state_num );
save('.\lkh_data\germany50_performance_struct_ddos1.mat','germany50_performance_struct_ddos1');

load('.\data\germany50_topo.mat');
load('.\data\germany50_sourcedata_ddos2.mat');
[germany50_performance_struct_ddos2]=AuxFun_CrossValidataion2_lkh( germany50_dst_entropy_ddos2, germany50_arrival_rate_ddos2, germany50_data_label_ddos2_block, germany50_adjacent_nodes_list, state_num );
save('.\lkh_data\germany50_performance_struct_ddos2.mat','germany50_performance_struct_ddos2');

state_num=3;
load('.\data\scalefree64_topo.mat');
load('.\data\scalefree64_sourcedata_ddos1_newV1.mat');
[scalefree64_performance_struct_ddos1]=AuxFun_CrossValidataion2_lkh( scalefree64_dst_entropy_ddos1, scalefree64_arrival_rate_ddos1, scalefree64_data_label_ddos1_block, scalefree64_adjacent_nodes_list, state_num );
save('.\lkh_data\scalefree64_performance_struct_ddos1.mat','scalefree64_performance_struct_ddos1');

state_num=3;
load('.\data\scalefree64_topo.mat');
load('.\data\scalefree64_sourcedata_ddos2_V1_add.mat');
[scalefree64_performance_struct_ddos2]=AuxFun_CrossValidataion2_lkh( scalefree64_dst_entropy_ddos2, scalefree64_arrival_rate_ddos2, scalefree64_data_label_ddos2_block, scalefree64_adjacent_nodes_list, state_num );
save('.\lkh_data\scalefree64_performance_struct_ddos2.mat','scalefree64_performance_struct_ddos2');



load('.\lkh_data\germany50_performance_struct_ddos1.mat');
lkh_germany50_ddos1=germany50_performance_struct_ddos1.likelihood_function_pertime;
load('.\lkh_data\germany50_performance_struct_ddos2.mat');
lkh_germany50_ddos2=germany50_performance_struct_ddos2.likelihood_function_pertime;

load('.\lkh_data\scalefree64_performance_struct_ddos1.mat');
lkh_scalefree64_ddos1=scalefree64_performance_struct_ddos1.likelihood_function_pertime;
load('.\lkh_data\scalefree64_performance_struct_ddos2.mat');
lkh_scalefree64_ddos2=scalefree64_performance_struct_ddos2.likelihood_function_pertime;

figure;
iteration=1:11;
nor_germany50_ddos1=mapminmax(lkh_germany50_ddos1(1,:),0,1); %取第一行，然后做归一化，最好看一下是哪一次的性能最好
plot(iteration,nor_germany50_ddos1);
hold on;
nor_lkh_germany50_ddos2=mapminmax(lkh_germany50_ddos2(1,:),0,1); %取第一行，然后做归一化
plot(iteration,nor_lkh_germany50_ddos2);
nor_lkh_scalefree64_ddos1=mapminmax(lkh_scalefree64_ddos1(1,:),0,1); %取第一行，然后做归一化
plot(iteration,nor_lkh_scalefree64_ddos1);
nor_lkh_scalefree64_ddos2=mapminmax(lkh_scalefree64_ddos2(1,:),0,1); %取第一行，然后做归一化
plot(iteration,nor_lkh_scalefree64_ddos2);

internet_norm_lkh=[nor_germany50_ddos1;nor_lkh_germany50_ddos2;nor_lkh_scalefree64_ddos1;nor_lkh_scalefree64_ddos2];
save('.\lkh_data\internet_norm_lkhV2.mat','internet_norm_lkh');


