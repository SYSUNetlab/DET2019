function [ class_series_km, centroid] = AuxFun_KmeansClustering( sample_data, class_num, replicates )
%   辅助函数：auxiliary function 对输入数据做kmeans聚类，特别地，是对一维数据的聚类，质心最小的作为第一类……
%   输入数据是样本，数据类别数，重复的次数
%******************************************************************************
if nargin < 3 
    replicates=10;
end
%根据kmeans划分，看看节点状态变化。
[class_series_km,centroid]=kmeans(sample_data,class_num,'Replicates',replicates);
%kmeans 得到的结果状态值跟均值不对应，有可能均值最大的是划分为状态1。因此这里需要转换一下。
temp_data1=[(1:class_num)',centroid];   
temp_data2=sortrows(temp_data1,2);   %按照第二列（质心）从小到大排列。
centroid=temp_data2(:,2);
temp_state_series_km=class_series_km;
for i=1:class_num
    class_series_km(temp_state_series_km==temp_data2(i,1))=i;   %状态1对应均值最小的, 更新idx
end

end