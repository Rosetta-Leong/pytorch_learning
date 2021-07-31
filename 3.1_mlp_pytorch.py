# -*-coding:utf-8-*-
# @File     :   2021/7/29 上午10:34
# @Author   :   Rosetta0
# @File     :   3.1_mlp_pytorch.py

import torch
from torch import nn
from d2l import torch as d2l

batch_size, lr, num_epochs = 256, 0.1, 10
#经测试pytoch实现版倒是epoch=10也没问题

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == "__main__":

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter,test_iter, loss, num_epochs, trainer)
    d2l.predict_ch3(net, test_iter)