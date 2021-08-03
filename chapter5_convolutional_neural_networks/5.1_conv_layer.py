# -*-coding:utf-8-*-
# @File     :   2021/8/3 下午3:08
# @Author   :   Rosetta0
# @File     :   5.1_conv_layer.py

import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape  #torch中是宽 * 高
    #输出矩阵形状
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

if __name__ == "__main__":

    # 二维互相关
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # print(corr2d(X, K))

    #利用卷积层简单的作边缘检测
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    # print(X)
    #卷积核 1*2
    #当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零。
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    # print(Y)
    #该卷积核仅能检测垂直边缘，如对输入进行转置就不行：
    # print(corr2d(X.t(), K))

    #已知输入输出，学习卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    #reshape为4D向量：通道， 批量大小， 宽， 高
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
        if (i+1)%2 == 0:
            print(f'batch {i+1}, loss {l.sum():.3f}')
    print(conv2d.weight.data.reshape(1,2))


