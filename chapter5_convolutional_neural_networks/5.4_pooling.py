# -*-coding:utf-8-*-
# @File     :   2021/8/5 上午9:38
# @Author   :   Rosetta0
# @File     :   5.4_pooling.py

import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

if __name__ == "__main__":

    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # print(pool2d(X, (2, 2)))
    # print(pool2d(X, (2, 2), 'avg'))

    #pytorch实现:默认情况下步幅与池化窗口大小相同（即下一块与上一块无重叠部分）
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    # print(X)
    pool2d = nn.MaxPool2d(3)    #3*3窗口
    # print(pool2d(X))

    #手动指定padding & stride
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    # print(pool2d(X))
    pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
    # print(pool2d(X))

    #在处理多通道输入数据时，[池化层在每个输入通道上单独运算]
    #这意味着池化层的输出通道数与输入通道数相同
    X = torch.cat((X, X+1), 1)
    print(X)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))    #输出通道数认为2
