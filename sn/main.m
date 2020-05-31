%% sw SI
clear variables;
state_num=2;
load('.\data\smallworld256_topo.mat');
load('.\data\smallworld256_sourcedata_SIV2.mat');
[smallworld256_performance_struct_SI]=AuxFun_CrossValidataion3( smallworld256_observedata_SI, smallworld256_label_SI_block, smallworld256_adjacent_nodes_list, state_num );
save('.\data1\smallworld256_performance_struct_SI_newcutV2.mat','smallworld256_performance_struct_SI');
temp_data=smallworld256_performance_struct_SI.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)

[performance_index_em,performance_em_pertime,confusion_matrix_em_pertime,em_estimate_state_series_block,parameter_em_pertime,likelihood_function_pertime]=AuxFun_SaveMatForPython(smallworld256_performance_struct_SI);
save('.\data2\smallworld256_performance_struct_SI_newcut_py_newV2.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');




%% sw SIS
clear variables;
state_num=2;
load('.\data\smallworld256_topo.mat');
load('.\data\smallworld256_sourcedata_SISV2.mat');
[smallworld256_performance_struct_SIS]=AuxFun_CrossValidataion3( smallworld256_observedata_SIS, smallworld256_label_SIS_block, smallworld256_adjacent_nodes_list, state_num );
save('.\data1\smallworld256_performance_struct_SIS_newcutV2.mat','smallworld256_performance_struct_SIS');
temp_data=smallworld256_performance_struct_SIS.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)

[performance_index_em,performance_em_pertime,confusion_matrix_em_pertime,em_estimate_state_series_block,parameter_em_pertime,likelihood_function_pertime]=AuxFun_SaveMatForPython(smallworld256_performance_struct_SIS);
save('.\data2\smallworld256_performance_struct_SIS_newcut_py_newV2.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');


%% sf SI
clear variables;
state_num=2;
load('.\data\freescale256_topo.mat');
load('.\data\freescale256_sourcedata_SIV2.mat');
[freescale256_performance_struct_SI]=AuxFun_CrossValidataion3( freescale256_observedata_SI, freescale256_label_SI_block, freescale256_adjacent_nodes_list, state_num);
save('.\data1\smallworld256_performance_struct_SI_newcutV2_newEst.mat','freescale256_performance_struct_SI');
temp_data=freescale256_performance_struct_SI.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)

[performance_index_em,performance_em_pertime,confusion_matrix_em_pertime,em_estimate_state_series_block,parameter_em_pertime,likelihood_function_pertime]=AuxFun_SaveMatForPython(freescale256_performance_struct_SI);
save('.\data2\freescale256_performance_struct_SI_newcut_py_newV2newEst.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');


%% sf SIS
clear variables;
state_num=2;
load('.\data\freescale256_topo.mat');
load('.\data\freescale256_sourcedata_SISV2.mat');
[freescale256_performance_struct_SIS]=AuxFun_CrossValidataion3( freescale256_observedata_SIS, freescale256_label_SIS_block, freescale256_adjacent_nodes_list, state_num);
save('.\data1\freescale256_performance_struct_SIS_newcut.mat','freescale256_performance_struct_SIS');
temp_data=freescale256_performance_struct_SIS.performance_em_pertime;
temp_mean1=mean(temp_data,2);
temp_mean2=mean(temp_mean1,3)

[performance_index_em,performance_em_pertime,confusion_matrix_em_pertime,em_estimate_state_series_block,parameter_em_pertime,likelihood_function_pertime]=AuxFun_SaveMatForPython(freescale256_performance_struct_SIS);
save('.\data2\freescale256_performance_struct_SIS_newcut_py_newV2.mat','performance_index_em','performance_em_pertime','confusion_matrix_em_pertime','em_estimate_state_series_block','parameter_em_pertime','likelihood_function_pertime');
