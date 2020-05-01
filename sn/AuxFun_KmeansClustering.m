function [ class_series_km, centroid] = AuxFun_KmeansClustering( sample_data, class_num, replicates )
%   ����������auxiliary function ������������kmeans���࣬�ر�أ��Ƕ�һά���ݵľ��࣬������С����Ϊ��һ�࡭��
%   ����������������������������ظ��Ĵ���
%******************************************************************************
if nargin < 3 
    replicates=10;
end
%����kmeans���֣������ڵ�״̬�仯��
[class_series_km,centroid]=kmeans(sample_data,class_num,'Replicates',replicates);
%kmeans �õ��Ľ��״ֵ̬����ֵ����Ӧ���п��ܾ�ֵ�����ǻ���Ϊ״̬1�����������Ҫת��һ�¡�
temp_data1=[(1:class_num)',centroid];   
temp_data2=sortrows(temp_data1,2);   %���յڶ��У����ģ���С�������С�
centroid=temp_data2(:,2);
temp_state_series_km=class_series_km;
for i=1:class_num
    class_series_km(temp_state_series_km==temp_data2(i,1))=i;   %״̬1��Ӧ��ֵ��С��, ����idx
end

end