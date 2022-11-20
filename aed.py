"""
# @Time    : 2022/9/6 18:12
# @File    : aed.py
# @Author  : rezheaiba
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import DenoisingAutoencoder
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

# global constants全局参数
BATCH_SIZE = 32
N_INP = 64
N_EPOCHS = 100
NOISE = 0.15


#  data loading数据加载
def custom_ts_multi_data_prep(dataset, start, window):  # 数据折叠
    X = []
    end = len(dataset)
    start = start + window
    for i in range(start, end):
        if i % window == 0:
            indices = range(i - window, i)
            X.append(dataset[indices].T)

    return np.array(X)


class noisedDataset(Dataset):

    def __init__(self, datasetclean):
        self.clean = datasetclean
        # self.transform = transform

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):  # 参数idx表示图片和标签在总数据集中的Index。

        xClean = self.clean[idx]

        return xClean


df = pd.read_csv(r'Metro-Interstate-Traffic-Volume-Encoded.csv')
x_scaler = preprocessing.MinMaxScaler()
dataX_temp = x_scaler.fit_transform(df[['traffic_volume']])
hist_window = 64
dataX = custom_ts_multi_data_prep(dataX_temp, 0, hist_window)
train_set = noisedDataset(dataX)  # tsfms) #datasetnoised, datasetclean, labels, transform

train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=False, drop_last=True)

running_loss = 0
epochloss = 0

l = len(train_loader)
auto_encoder = DenoisingAutoencoder()
optimizer = optim.Adam(auto_encoder.parameters(), lr=1e-3)
losslist = list()
criterion = nn.MSELoss()
# 开始训练
for epoch in range(N_EPOCHS):
    for (x, z) in enumerate(train_loader):
        y = z.view(z.size()[0], -1)
        # 从非空序列中随机选取一个数据并返回，该序列可以是list、tuple、str、set
        noise = np.random.choice([1, 0], size=(BATCH_SIZE, N_INP), p=[NOISE, 1 - NOISE])
        inp = y * torch.FloatTensor(noise)
        inp = inp.float()
        y = y.float()
        mid, decoded = auto_encoder(inp)
        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epochloss += loss.item()
    losslist.append(running_loss / l)
    running_loss = 0
    print("======> epoch: {}/{}, Loss:{}".format(epoch, N_EPOCHS, loss.item()))

t_new_sum = np.empty((1, 0))
with torch.no_grad():
    for (x, z) in enumerate(train_loader):
        y = z.view(z.size()[0], -1)
        # noise = np.random.choice([1, 0], size=(BATCH_SIZE, N_INP), p=[NOISE, 1 - NOISE])
        # inp = z * torch.FloatTensor(noise)
        inp = y.float()
        mid, decoded = auto_encoder(inp)
        decoded = decoded.view(1, -1)
        output = decoded.detach().cpu().numpy()
        t_new_sum = np.hstack((t_new_sum, output))  # np.hstack将参数元组的元素数组按水平方向进行叠加

plt.plot(range(len(losslist)), losslist)
plt.show()