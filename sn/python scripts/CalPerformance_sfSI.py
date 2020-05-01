import numpy as np
import time
import scipy.io as mode1o
from sklearn import cluster, mixture
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def re_label(source_data, label, label_num):
    # label 是一维的数据
    label_mean_mat = np.arange(2 * label_num).reshape(2, label_num).astype(float)
    (sample_num,) = label.shape
    new_label = np.zeros((sample_num,))
    for i in range(label_num):
        idx = np.nonzero(label == i)
        temp_data = source_data[idx[0], :]  # 一维数据
        temp_mean = np.mean(temp_data)
        label_mean_mat[1, i] = temp_mean
    label_mean_mat = label_mean_mat[:, label_mean_mat[1].argsort()]

    for i in range(label_num):
        new_label_idx = label_mean_mat[0, i]
        idx = np.nonzero(label == new_label_idx)
        new_label[idx[0]] = i + 1
    return new_label  # 输出是一维数组


if __name__ == "__main__":
    time1 = time.time()
    folds = 10  # split source data into 10 folds, 10 folds cross validation
    test_fold_num = 1  # 每次用于test的fold数量
    repeat_times = 10  # 重复十折交叉验证的次数
    # load mat files

    mat_files = ['./data/freescale256_sourcedata_SIV2.mat']
    sourcedata_names = ['freescale256_observedata_SI']
    label_names = ['freescale256_label_SI_block']
    parameters = {'n_clusters': 2}

    for file_list_id in range(len(mat_files)):
        current_mat_file = mode1o.loadmat(mat_files[file_list_id])  # <class 'dict'>
        sourcedata_block = current_mat_file[sourcedata_names[file_list_id]]
        label_block = current_mat_file[label_names[file_list_id]]  # <class 'numpy.ndarray'>
        (node_num, timeslot) = sourcedata_block.shape
        timeslot_perfold = int(timeslot / folds)
        print(node_num, '\t', timeslot_perfold)

        # 预先分配资源给变量
        estimate_state_series_block_KMeans = np.zeros((node_num * test_fold_num * timeslot_perfold, folds * repeat_times))
        estimate_state_series_block_GMM = np.zeros((node_num * test_fold_num * timeslot_perfold, folds * repeat_times))
        estimate_state_series_block_Ward = np.zeros((node_num * test_fold_num * timeslot_perfold, folds * repeat_times))
        estimate_state_series_block_Birch = np.zeros((node_num * test_fold_num * timeslot_perfold, folds * repeat_times))
        performance_pertime_train_KMeans = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_test_KMeans = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_train_GMM = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_test_GMM = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_train_Ward = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_test_Ward = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_train_Birch = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        performance_pertime_test_Birch = np.zeros((folds * repeat_times, 5, parameters['n_clusters']))
        confusion_matrix_pertime_KMeans = np.zeros((folds * repeat_times, parameters['n_clusters'], parameters['n_clusters']))
        confusion_matrix_pertime_GMM = np.zeros((folds * repeat_times, parameters['n_clusters'], parameters['n_clusters']))
        confusion_matrix_pertime_Ward = np.zeros((folds * repeat_times, parameters['n_clusters'], parameters['n_clusters']))
        confusion_matrix_pertime_Birch = np.zeros((folds * repeat_times, parameters['n_clusters'], parameters['n_clusters']))

        estimate_state_series_block = np.zeros((node_num * timeslot_perfold, folds))
        performance_pertime_train = np.zeros((5, parameters['n_clusters']))
        performance_pertime_test = np.zeros((5, parameters['n_clusters']))
        confusion_matrix_pertime = np.zeros((parameters['n_clusters'], parameters['n_clusters']))

        for repeat_id in range(repeat_times):
            # shuffle
            # 这里首先对数据进行shuffle操作，将顺序乱序。
            shuffle_index = np.arange(0, timeslot)
            np.random.shuffle(shuffle_index)
            sourcedata_block = sourcedata_block[:, shuffle_index]
            label_block = label_block[:, shuffle_index]

            # 为了方便索引，这里需要对数据复制
            sourcedata_block = np.tile(sourcedata_block, (1, 2))
            label_block = np.tile(label_block, (1, 2))

            for fold_id in range(folds):
                test_index = np.arange(fold_id * timeslot_perfold, (fold_id + test_fold_num) * timeslot_perfold).astype(np.int)
                train_index = np.arange((fold_id + test_fold_num) * timeslot_perfold, (fold_id + 10) * timeslot_perfold).astype(np.int)
                # temp_index_set = np.arange(fold_id * timeslot_perfold, (fold_id + 10) * timeslot_perfold)
                train_data = sourcedata_block[:, train_index]
                train_label = label_block[:, train_index]
                test_data = sourcedata_block[:, test_index]
                test_label = label_block[:, test_index]
                # 将所有时隙的数据给连接起来
                # 在Python中reshape是按照行来变换的，与MATLAB不同，因此需要先做转置
                train_data = train_data.transpose()
                train_data = train_data.reshape(-1, 1)
                train_label = train_label.transpose()
                train_label = train_label.reshape(-1, 1)
                test_data = test_data.transpose()
                test_data = test_data.reshape(-1, 1)
                test_label = test_label.transpose()
                test_label = test_label.reshape(-1, 1)

                # create cluster object
                kmeans = cluster.KMeans(n_clusters=parameters['n_clusters'])
                gmm = mixture.GaussianMixture(n_components=parameters['n_clusters'], covariance_type='full', max_iter=20, random_state=0)
                ward = cluster.AgglomerativeClustering(n_clusters=parameters['n_clusters'], linkage='ward')
                birch = cluster.Birch(n_clusters=parameters['n_clusters'])

                clustering_algorithm = (
                    ('KMeans', kmeans),
                    ('GaussianMixture', gmm),
                    ('Ward', ward),
                    ('Birch', birch)
                )

                for name, algorithm in clustering_algorithm:
                    t0 = time.time()
                    # estimator = algorithm.fit(train_data)

                    if name == 'Ward':
                        # algorithm.fit(train_data)
                        # train_pre = algorithm.labels_.astype(np.int)
                        train_pre = train_label - 1  # 因为使用层次聚类的耗时比较多，因此这里对于训练数据不重新聚类
                        # train_pre[0:100, 0] = 1  # 引入一些噪声
                        # train_pre[100:200, 0] = 2
                        train_pre = np.reshape(train_pre, (-1,))

                        # test_data = test_data[0:node_num * timeslot_perfold, 0]  # 只取出一折来计算
                        # test_data = np.reshape(test_data, (-1, 1))

                        algorithm.fit(test_data)
                        test_pre = algorithm.labels_.astype(np.int)
                    else:
                        algorithm.fit(train_data)
                        train_pre = algorithm.predict(train_data)
                        test_pre = algorithm.predict(test_data)

                    # 对预测出来的label换标号，均值小的为1，均值大的为2，因为pre之后得到的label是0和1。
                    train_pre = re_label(source_data=train_data, label=train_pre, label_num=parameters['n_clusters'])
                    test_pre = re_label(source_data=test_data, label=test_pre, label_num=parameters['n_clusters'])

                    # train_accuracy = np.mean(train_pre.ravel() == train_label.ravel()) * 100
                    # test_accuracy = np.mean(test_pre.ravel() == test_label.ravel()) * 100

                    accuracy_train = accuracy_score(train_label, train_pre)
                    error_train = 1 - accuracy_train
                    performance_pertime_train[0, :] = accuracy_train
                    performance_pertime_train[1, :] = error_train
                    performance_pertime_train[2, :] = precision_score(train_label, train_pre, average=None)
                    performance_pertime_train[3, :] = recall_score(train_label, train_pre, average=None)
                    performance_pertime_train[4, :] = f1_score(train_label, train_pre, average=None)
                    print(name, 'performance_pertime_train in ', repeat_id * repeat_times + fold_id, 'th times \n', performance_pertime_train)
                    print(classification_report(train_label, train_pre))

                    accuracy_test = accuracy_score(test_label, test_pre)
                    error_test = 1 - accuracy_test
                    performance_pertime_test[0, :] = accuracy_test
                    performance_pertime_test[1, :] = error_test
                    performance_pertime_test[2, :] = precision_score(test_label, test_pre, average=None)
                    performance_pertime_test[3, :] = recall_score(test_label, test_pre, average=None)
                    performance_pertime_test[4, :] = f1_score(test_label, test_pre, average=None)
                    print(name, 'performance_pertime_test in ', repeat_id * repeat_times + fold_id, 'th times \n', performance_pertime_test)
                    print(classification_report(test_label, test_pre))

                    confusion_matrix_pertime = confusion_matrix(test_label, test_pre)

                    # 这里使用if语句判断是哪一种方法，然后对应不用的方法，存储不同的性能参数。
                    if name == 'KMeans':
                        performance_pertime_train_KMeans[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_train
                        performance_pertime_test_KMeans[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_test
                        estimate_state_series_block_KMeans[:, repeat_id * repeat_times + fold_id] = test_pre
                        confusion_matrix_pertime_KMeans[repeat_id * repeat_times + fold_id, :, :] = confusion_matrix_pertime
                    elif name == 'GaussianMixture':
                        performance_pertime_train_GMM[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_train
                        performance_pertime_test_GMM[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_test
                        estimate_state_series_block_GMM[:, repeat_id * repeat_times + fold_id] = test_pre
                        confusion_matrix_pertime_GMM[repeat_id * repeat_times + fold_id, :, :] = confusion_matrix_pertime
                    elif name == 'Ward':
                        performance_pertime_train_Ward[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_train
                        performance_pertime_test_Ward[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_test
                        estimate_state_series_block_Ward[:, repeat_id * repeat_times + fold_id] = test_pre
                        confusion_matrix_pertime_Ward[repeat_id * repeat_times + fold_id, :, :] = confusion_matrix_pertime
                    else:
                        performance_pertime_train_Birch[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_train
                        performance_pertime_test_Birch[repeat_id * repeat_times + fold_id, :, :] = performance_pertime_test
                        estimate_state_series_block_Birch[:, repeat_id * repeat_times + fold_id] = test_pre
                        confusion_matrix_pertime_Birch[repeat_id * repeat_times + fold_id, :, :] = confusion_matrix_pertime
                    t1 = time.time()
                    print(name, " elapsed time :", t1 - t0)
        # average performance
        performance_average_train_KMeans = np.mean(performance_pertime_train_KMeans, 0)
        performance_average_test_KMeans = np.mean(performance_pertime_test_KMeans, 0)
        performance_average_train_GMM = np.mean(performance_pertime_train_GMM, 0)
        performance_average_test_GMM = np.mean(performance_pertime_test_GMM, 0)
        performance_average_train_Ward = np.mean(performance_pertime_train_Ward, 0)
        performance_average_test_Ward = np.mean(performance_pertime_test_Ward, 0)
        performance_average_train_Birch = np.mean(performance_pertime_train_Birch, 0)
        performance_average_test_Birch = np.mean(performance_pertime_test_Birch, 0)

        freescale256_SI_performance = {'performance_average_train_KMeans': performance_average_train_KMeans,
                                       'performance_average_test_KMeans': performance_average_test_KMeans,
                                       'performance_pertime_train_KMeans': performance_pertime_train_KMeans,
                                       'performance_pertime_test_KMeans': performance_pertime_test_KMeans,
                                       'estimate_state_series_block_KMeans': estimate_state_series_block_KMeans,
                                       'confusion_matrix_pertime_KMeans': confusion_matrix_pertime_KMeans,
                                       'performance_average_train_GMM': performance_average_train_GMM,
                                       'performance_average_test_GMM': performance_average_test_GMM,
                                       'performance_pertime_train_GMM': performance_pertime_train_GMM,
                                       'performance_pertime_test_GMM': performance_pertime_test_GMM,
                                       'estimate_state_series_block_GMM': estimate_state_series_block_GMM,
                                       'confusion_matrix_pertime_GMM': confusion_matrix_pertime_GMM,
                                       'performance_average_train_Ward': performance_average_train_Ward,
                                       'performance_average_test_Ward': performance_average_test_Ward,
                                       'performance_pertime_train_Ward': performance_pertime_train_Ward,
                                       'performance_pertime_test_Ward': performance_pertime_test_Ward,
                                       'estimate_state_series_block_Ward': estimate_state_series_block_Ward,
                                       'confusion_matrix_pertime_Ward': confusion_matrix_pertime_Ward,
                                       'performance_average_train_Birch': performance_average_train_Birch,
                                       'performance_average_test_Birch': performance_average_test_Birch,
                                       'performance_pertime_train_Birch': performance_pertime_train_Birch,
                                       'performance_pertime_test_Birch': performance_pertime_test_Birch,
                                       'estimate_state_series_block_Birch': estimate_state_series_block_Birch,
                                       'confusion_matrix_pertime_Birch': confusion_matrix_pertime_Birch}

    with open('./result/performance_freescale256_SIV2.pickle', 'wb') as f:
        pickle.dump([freescale256_SI_performance], f)

    with open('./result/performance_freescale256_SIV2.pickle', 'rb') as f:
        freescale256_SI_performance = pickle.load(f)
    time2 = time.time()
    print('total elapsed time: ', time2 - time1)
