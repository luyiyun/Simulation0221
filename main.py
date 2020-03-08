from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from simudata import SimulateData
from AE import AEEstimator
from evaluation import cluster


def main():
    infor_sample_num = 20
    other_sample_num = 1000

    label_var_num = 5
    infor_var_num = 5
    other_var_num = 1000

    label_diff = 1
    infor_diff = 0.5

    std = 1

    n_cluster = 2
    n_comp = 10
    cluster_method = "kmeans"

    # 得到模拟数据
    data = SimulateData(
        infor_sample_num, other_sample_num, label_var_num,
        infor_var_num, other_var_num, label_diff=label_diff,
        infor_diff=infor_diff, std=std
    )
    X, y = data.generate()

    # 使用原始数据来直接进行聚类
    ori_pred = cluster(X, n_cluster, cluster_method)
    ori_score = adjusted_rand_score(data.infor_label, ori_pred)

    # 使用PCA转换数据后再进行聚类
    pca_est = PCA(n_comp)
    pca_trans = pca_est.fit_transform(X)
    pca_pred = cluster(pca_trans, n_cluster, cluster_method)
    pca_score = adjusted_rand_score(data.infor_label, pca_pred)

    # 使用AE，取其中0.2的样本作为valid
    ae_est = AEEstimator(data.shape[1], [500, 200], n_comp, 0.01, 64, 100)
    ae_est.fit(X, 0.2)
    ae_trans = ae_est.transform(X)
    ae_pred = cluster(ae_trans, n_cluster, cluster_method)
    ae_score = adjusted_rand_score(data.infor_label, ae_pred)

    print(ori_score, pca_score, ae_score)


if __name__ == "__main__":
    main()
