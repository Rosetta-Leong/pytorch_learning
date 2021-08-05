# -*-coding:utf-8-*-
# @File     :   2021/8/3 下午4:11
# @Author   :   Rosetta0
# @File     :   5.2_padding-stride.py

import torch
from torch import nn
from d2l import torch as d2l

def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

if __name__ == "__main__":

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 上下左右各填充一行（列）, 与理论公式有区别，代入计算时要乘2
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)

    #不对称padding
    conv2d = nn.Conv2d(1, 1, kernel_size=(5,3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)

    #将高度与宽度的步幅设置为2
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape) #此时向下取整到 2 * 2