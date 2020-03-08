from sklearn.cluster import KMeans


# 聚类方法
def cluster(X, n_cluster=10, method="kmeans"):
    method_map = {
        "kmeans": KMeans
    }
    estimator = method_map[method](n_cluster)
    return estimator.fit_predict(X)
