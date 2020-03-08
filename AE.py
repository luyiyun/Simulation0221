from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Loss:

    def __init__(self):
        self.sum = 0.
        self.count = 0

    def add(self, loss, batch):
        self.sum += loss.cpu().item() * batch
        self.count += batch

    def value(self):
        return self.sum / self.count


class AEModel(nn.Module):

    def __init__(self, inSize, hidden_units, bottle_neck_units):
        super().__init__()
        encode_units = [inSize] + hidden_units + [bottle_neck_units]
        decode_units = [bottle_neck_units] + hidden_units[::-1] + [inSize]

        self.encode = []
        for i, (it, ot) in enumerate(zip(encode_units[:-1], encode_units[1:])):
            self.encode.append(nn.Linear(it, ot))
            if i < (len(encode_units) - 2):
                self.encode.append(nn.BatchNorm1d(ot))
                self.encode.append(nn.ReLU())
        self.encode = nn.Sequential(*self.encode)

        self.decode = []
        for i, (it, ot) in enumerate(zip(decode_units[:-1], decode_units[1:])):
            self.decode.append(nn.Linear(it, ot))
            if i < (len(decode_units) - 2):
                self.decode.append(nn.BatchNorm1d(ot))
                self.decode.append(nn.ReLU())
        self.decode = nn.Sequential(*self.decode)

    def forward(self, x):
        return self.decode(self.encode(x))

    def Encode(self, x):
        return self.encode(x)


class AEEstimator:
    def __init__(
        self, inSize, hidden_units, bottle_neck_units,
        lr, bs, epoch, device="cuda:0", seed=1234
    ):
        self.lr = lr
        self.bs = bs
        self.epoch = epoch
        self.device = device
        self.seed = seed
        self.AEmodel = AEModel(
            inSize, hidden_units, bottle_neck_units).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.AEmodel.parameters(), lr=lr)
        self.history = {}

    def fit(self, X, test=None):
        # 整理数据集
        if isinstance(test, float):
            if test > 1:
                raise ValueError("test size must be lower than 1.")
            train_arr, test_arr = train_test_split(
                X, test_size=test, random_state=self.seed)
        elif isinstance(test, np.ndarray):
            if test.shape[1] != X.shape[1]:
                raise ValueError("The shape[1] of test must be same as X.")
            train_arr, test_arr = X, test
        elif test is None:
            train_arr, test_arr = X, None
        else:
            raise ValueError("test must be ndarray or float.")
        datasets = {
            "train": data.TensorDataset(torch.tensor(train_arr).float()),
        }
        if test_arr is not None:
            datasets["test"] = data.TensorDataset(
                torch.tensor(test_arr).float()
            )
        dataloaders = {
            k: data.DataLoader(v, self.bs, k == "train")
            for k, v in datasets.items()
        }
        # 整理训练是需要记录的内容
        self.history["loss"] = {k: [] for k in dataloaders.keys()}
        self.best = {
            "model": deepcopy(self.AEmodel.state_dict()),
            "epoch": -1,
            "loss": np.inf
        }
        # 训练
        for e in tqdm(range(self.epoch), "Epoch:"):
            for phase in dataloaders.keys():
                loss_obj = Loss()
                for bs_x, in tqdm(dataloaders[phase], phase + ":"):
                    bs_x = bs_x.to(self.device)
                    with torch.set_grad_enabled(phase == "train"):
                        recon = self.AEmodel(bs_x)
                        loss = self.criterion(recon, bs_x)
                        loss_obj.add(loss, bs_x.size(0))
                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                self.history["loss"][phase].append(loss_obj.value())
            if "test" in self.history["loss"].keys():
                now_loss = self.history["loss"]["test"][-1]
                if now_loss < self.best["loss"]:
                    self.best["epoch"] = e
                    self.best["model"] = deepcopy(self.AEmodel.state_dict())
                    self.best["loss"] = now_loss

        # 训练完成后将最好的一次参数load
        if self.best["epoch"] > 0:
            self.AEmodel.load_state_dict(self.best["model"])

    def transform(self, X):
        dataset = data.TensorDataset(torch.tensor(X).float())
        dataloader = data.DataLoader(dataset, self.bs, False)
        recons = []
        for bs_x, in tqdm(dataloader):
            bs_x = bs_x.to(self.device)
            with torch.no_grad():
                recon = self.AEmodel.Encode(bs_x)
                recons.append(recon)
        recons = torch.cat(recons, dim=0)
        return recons.cpu().numpy()


if __name__ == "__main__":
    X = np.random.rand(100, 10)
    test = np.random.rand(100, 10)
    estimator = AEEstimator(10, [5, 5], 2, 0.01, 64, 100)
    estimator.fit(X, test)
    features = estimator.transform(X)
    print(features)
    print(estimator.history)
    print(estimator.best["epoch"])
    print(estimator.best["loss"])