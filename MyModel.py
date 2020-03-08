import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -(- (U + eps).log() + eps).log()


def gumbel_softmax(logits, temperature):
    y = logits + sample_gumbel(logits.shape).to(logits)
    return F.softmax(y / temperature, dim=-1)


class MyModel(nn.Module):

    def __init__(self, inSize, temp, hard=True):
        super().__init__()
        self.temp = temp
        self.hard = hard
        self.InforDisc = nn.Sequential(
            nn.Linear(inSize, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.LogSoftmax(dim=-1)
        )
        self.LabelDisc = nn.Sequential(
            nn.Linear(inSize, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        zlog = self.InforDisc(x)
        ylogit = self.LabelDisc(x)
        zsample = gumbel_softmax(zlog, self.temp)
        if self.hard:
            zsample, _ = zsample.max(-1)
        else:
            zsample = zsample[:, -1]
        return ylogit, zsample

    def predict(self, x, type="infor"):
        if type == "infor":
            return self.InforDisc(x).exp()
        elif type == "label":
            return F.sigmoid(self.LabelDisc(x))


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y, ylogit, zsample):
        losses = self.loss_func(ylogit.squeeze(), y)
        return (losses * (2 * zsample - 1)).sum()


class MyEstimator:
    def __init__(self, insize, temp, lr, bs, epoch, device="cuda:0"):

        self.lr = lr
        self.bs = bs
        self.epoch = epoch
        self.device = device
        self.model = MyModel(insize, temp).to(device)
        self.criterion = MyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y):

        dataset = data.TensorDataset(torch.tensor(X).float(),
                                     torch.tensor(y).float())
        dataloader = data.DataLoader(dataset, self.bs, True)

        for e in tqdm(range(self.epoch)):
            for bs_x, bs_y in tqdm(dataloader):
                bs_x = bs_x.to(self.device)
                bs_y = bs_y.to(self.device)
                with torch.enable_grad():
                    self.optimizer.zero_grad()
                    ylogit, zsample = self.model(bs_x)
                    loss = self.criterion(bs_y, ylogit, zsample)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        dataset = data.TensorDataset(torch.tensor(X).float())
        dataloader = data.DataLoader(dataset, self.bs, False)
        predicts = []
        for bs_x, in tqdm(dataloader):
            bs_x = bs_x.to(self.device)
            with torch.no_grad():
                predict = self.model.predict(bs_x, "infor")
                predicts.append(predict)
        predicts = torch.cat(predicts, dim=0)
        return predicts.cpu().numpy()
