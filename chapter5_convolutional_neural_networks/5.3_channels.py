# -*-coding:utf-8-*-
# @File     :   2021/8/4 下午4:26
# @Author   :   Rosetta0
# @File     :   5.3_channels.py

import torch
from d2l import torch as d2l

#多通道输入互相关运算
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
    #使用 zip() 函数“压缩”多个序列时，它会分别取各序列中第 1 个元素、第 2 个元素、... 第 n 个元素，各自组成新的元组
    #此处相当于先取第一个通道与其对应二维卷积核做互相关，然后依次是第二个...

#多通道输入、输出
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

#1 * 1卷积(利用全连接实现)
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == "__main__":
    #多输入单输出
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    # print(corr2d_multi_in(X, K))

    #多输入多输出
    K = torch.stack((K, K + 1, K + 2), 0)
    # print(K.shape)
    # print(K)
    # print(corr2d_multi_in_out(X, K))

    # 1 * 1卷积
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6