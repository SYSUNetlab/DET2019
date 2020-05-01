import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 读取多次实验性的能平均值
    mat_files = ['./result/scalefree64_performance_struct_ddos1_py_newV1.mat']
    performance_GNB2A = []
    for i in range(len(mat_files)):
        current_mat_file = sio.loadmat(mat_files[i])  # current_mat_file 是一个字典
        performance_GNB2A.append(current_mat_file['performance_em_pertime'])  # 每一个元素都是5*3*100的ndarray

    with open('./result/performance_scalefree64_ddos1_newV1.pickle', 'rb') as f:
        germany50_ddos1_performance = pickle.load(f)
    germany50_ddos1_performance=germany50_ddos1_performance[0]
    print(germany50_ddos1_performance['performance_pertime_test_KMeans'])
    # performance_others = [colt_ddos1_performance, colt_ddos2_performance, smallworld128_ddos1_performance, smallworld128_ddos2_performance]  # 每一个元素都是一个字典。
    performance_KMeans = [germany50_ddos1_performance['performance_pertime_test_KMeans']]  # 每一个元素都是 100*5*3的ndarray
    performance_GMM = [germany50_ddos1_performance['performance_pertime_test_GMM']]  # 每一个元素都是 100*5*3的ndarray
    performance_Ward = [germany50_ddos1_performance['performance_pertime_test_Ward']]  # 每一个元素都是 100*5*3的ndarray
    performance_Birch = [germany50_ddos1_performance['performance_pertime_test_Birch']]  # 每一个元素都是 100*5*3的ndarray

    performance_others = [performance_Birch, performance_Ward, performance_KMeans, performance_GMM]

    eps_prefix = './performance_data/'
    eps_name = [eps_prefix + 'scalefree64_performance_ddos1.eps']
    figure_title = ['scalefree64 ddos1']
    avg_performance = []  # 性能的平均值

    # 因为坐标轴的刻度不一致，所以需要一个一个地画
    # ******************************************************************************************************
    i = 0
    performance_avg_GNB2A = np.mean(performance_GNB2A[i], axis=1)  # 5*100的ndarray
    performance_avg_others = []
    for j in range(len(performance_others)):
        performance_avg_others.append(np.mean(performance_others[j][i], axis=2))  # 每一个元素都是100*5
    accuracy_all = []
    macro_f1_all = []
    for j in range(len(performance_others)):
        accuracy_all.append(100 * performance_avg_others[j][:, 0])
        macro_f1_all.append(100 * performance_avg_others[j][:, 4])
    accuracy_all.append(100 * performance_avg_GNB2A[0, :])  # *100表示转换成百分数
    macro_f1_all.append(100 * performance_avg_GNB2A[4, :])
    # 开始画图
    fig, left_ax = plt.subplots(figsize=(6, 6))
    right_ax = left_ax.twinx()
    box_scale = 0.7
    box_left = left_ax.get_position()
    left_ax.set_position([box_left.x0, box_left.y0, box_left.width * box_scale, box_left.height])
    box_right = right_ax.get_position()
    right_ax.set_position([box_right.x0, box_right.y0, box_right.width * box_scale, box_right.height])

    left_ax.grid(True, axis='y')
    # right_ax.grid(True, axis='y')
    medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick')  # median线的特性
    bplot1 = left_ax.boxplot(accuracy_all, positions=range(1, 6), notch=False, sym='s', vert=True, patch_artist=True, showfliers=False, medianprops=medianprops)
    bplot2 = right_ax.boxplot(macro_f1_all, positions=range(7, 12), notch=False, sym='s', vert=True, patch_artist=True, showfliers=False, medianprops=medianprops)
    left_ax.plot([6, 6], [25, 100], color='black', linewidth=1, linestyle='dashed')  # 在两种性能中间画一条区分线
    left_ax.set_xticks([3, 9])
    left_ax.set_xticklabels(['Accuracy', 'Macro-F1'], fontsize=15)
    left_ax.set_ylabel('Values (%)', fontsize=15)
    left_ax.tick_params(axis='y', labelsize=12)
    right_ax.set_ylabel('Values (%)', fontsize=15)
    right_ax.tick_params(axis='y', labelsize=12)
    right_ax.set_title(figure_title[i], fontsize=15)

    left_ax.set_xlim(0, 12)
    left_ax.set_ylim(60, 90)
    right_ax.set_xlim(0, 12)
    right_ax.set_ylim(30, 95)

    # fill with colors
    colors = ['orange', 'pink', 'lightblue', 'lightgreen', 'grey']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    left_ax.legend([bplot1["boxes"][0], bplot1["boxes"][1], bplot1["boxes"][2], bplot1["boxes"][3], bplot1["boxes"][4]], ['Birch', 'Ward', 'KMeans', 'GMM', 'STCA'], loc='upper left',
                   bbox_to_anchor=(1.1, 1), fontsize=12)

    #plt.savefig(eps_name[i], format="eps")
    # plt.show()

    # 记录平均值
    avg_performance_temp = np.zeros((5, 5))  # 每一行记录每一种方法的平均性能，性能指标包括 准确率、错误率、精准率、召回率、F1
    for j in range(len(performance_others)):
        avg_performance_temp[j, :] = np.mean(performance_avg_others[j], axis=0)  # performance_avg_others 的每一个元素是100*5的ndarray类型数据
    avg_performance_temp[4, :] = np.mean(performance_avg_GNB2A, axis=1)  # performance_avg_GNB2A是 5*100的ndarray
    avg_performance.append(avg_performance_temp)
    # ******************************************************************************************************

    print('ok')
    print(avg_performance)
    for i in range(1):
        print(avg_performance[i])

    plt.show()
