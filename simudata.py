import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from MyModel import MyEstimator
from AE import AEEstimator


class SimulateData:

    def __init__(
        self, infor_sample_num, other_sample_num, label_var_num,
        infor_var_num, other_var_num, label_diff=1, infor_diff=1,
        std=1
    ):
        self.infor_sample_num = infor_sample_num
        self.other_sample_num = other_sample_num
        self.label_var_num = label_var_num
        self.infor_var_num = infor_var_num
        self.other_var_num = other_var_num
        self.label_diff = label_diff
        self.infor_diff = infor_diff
        self.std = std

    def generate(self, rand_seed=1234):
        np.random.seed(rand_seed)
        # infor
        label_var_means = np.random.normal(size=self.label_var_num)
        part1 = np.concatenate([
            np.random.normal(label_var_means-self.label_diff/2, self.std,
                             size=(self.infor_sample_num, self.label_var_num)),
            np.random.normal(label_var_means+self.label_diff/2, self.std,
                             size=(self.infor_sample_num, self.label_var_num)),
            np.random.normal(scale=self.std,
                             size=(self.other_sample_num, self.label_var_num))
        ])
        # part2
        part2_var_means = np.random.normal(scale=self.std,
                                           size=self.infor_var_num)
        part2 = np.concatenate([
            np.random.normal(
                part2_var_means-self.infor_diff/2, self.std,
                size=(self.infor_sample_num*2, self.infor_var_num)
            ),
            np.random.normal(
                part2_var_means+self.infor_diff, self.std,
                size=(self.other_sample_num, self.infor_var_num)
            )
        ])
        # part3
        part3 = np.random.normal(
            scale=self.std,
            size=(self.infor_sample_num*2+self.other_sample_num,
                  self.other_var_num)
        )
        # X
        X = np.concatenate([part1, part2, part3], axis=1)
        # Y
        y = np.r_[
            np.zeros(self.infor_sample_num), np.ones(self.infor_sample_num),
            np.random.randint(2, size=self.other_sample_num)
        ]
        # isInfor
        return X, y

    @property
    def shape(self):
        return (self.infor_sample_num * 2 + self.other_sample_num,
                self.label_var_num + self.infor_var_num + self.other_var_num)

    @property
    def infor_label(self):
        return np.r_[
            np.ones(self.infor_sample_num*2), np.zeros(self.other_sample_num)
        ]


class SeparateSamples:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, label_y):
        self.estimator.fit(X, y)

    def predict(self, X):
        prob = self.estimator.predict_proba(X)
        return np.abs(prob - 0.5)

    def fit_predict(self, X, label_y):
        self.fit(X, label_y)
        return self.predict(X)

    def score(self, X, infor_y):
        prob = self.predict(X)[:, 1]
        return roc_auc_score(infor_y, prob)


if __name__ == "__main__":
    infor_sample_num = 20
    other_sample_num = 1000

    label_var_num = 5
    infor_var_num = 5
    other_var_num = 1000

    label_diff = 1
    infor_diff = 0.5

    std = 1

    # 得到模拟数据
    data = SimulateData(
        infor_sample_num, other_sample_num, label_var_num,
        infor_var_num, other_var_num, label_diff=label_diff,
        infor_diff=infor_diff, std=std
    )
    X, y = data.generate()

    # 对此数据进行10-CV，使用预测器是RF和logistic
    if False:
        estimators = {
            "logistic": LogisticRegression(solver="lbfgs"),
            "RF": RandomForestClassifier(n_estimators=500)
        }
        for k, v in estimators.items():
            scores = cross_val_score(
                v, X, y, scoring="roc_auc", n_jobs=-1, cv=10)
            print("estimator: %s, mean of scores: %.4f" % (k, np.mean(scores)))

    # 使用朴素的思想来进行看看效果
    if False:
        for k, v in estimators.items():
            seperate_model = SeparateSamples(v)
            seperate_model.fit(X, y)
            res = seperate_model.score(X, data.infor_label)
            print("estimator: %s, res: %.4f" % (k, res))

    # 主成分分析看一下
    if False:
        pca = PCA(2)
        pca_x = pca.fit_transform(X)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(
            pca_x[:infor_sample_num, 0], pca_x[:infor_sample_num, 1], 'ro'
        )
        ax.plot(
            pca_x[infor_sample_num:infor_sample_num*2, 0],
            pca_x[infor_sample_num:infor_sample_num*2, 1], 'bo'
        )
        ax.plot(
            pca_x[infor_sample_num*2:, 0], pca_x[infor_sample_num*2:, 1], 'go'
        )
        fig.savefig("./RESULTS/pca.png")
        plt.close(fig)

    # 高斯混合模型做一下
    if False:
        gau_mix = GaussianMixture(2)
        pred = gau_mix.fit_predict(X)
        print(pd.crosstab(pred, data.infor_label))

    # kmeans
    if False:
        kmeans = KMeans(2)
        pred = kmeans.fit_predict(X)
        print(pd.crosstab(pred, data.infor_label))

    # 自编深度
    if False:
        my_est = MyEstimator(data.shape[1], 0.1, 0.01, 64, 100)
        my_est.fit(X, y)
        pred = my_est.predict(X)
        print(pred)

    # AE
    if False:
        ae_est = AEEstimator(data.shape[1], [500, 250], 10, 0.01, 64, 100)
        ae_est.fit(X)
        encodes = ae_est.transform(X)
