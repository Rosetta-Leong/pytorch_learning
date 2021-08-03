# -*-coding:utf-8-*-
# @File     :   2021/8/1 下午2:45
# @Author   :   Rosetta0
# @File     :   4.3_custom_layer.py
# 自定义层，同样继承于nn.Module

import torch
import torch.nn.functional as F
from torch import nn

#无参数
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


#有参数
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))    #参数应为nn.Parameter的示例
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear =torch.matmul(X, self.weight.data) + self.bias.data
        return  F.relu(linear)



if __name__ == "__main__":

    #单独使用自定义层
    # layer = CenteredLayer()
    # print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
    #将自定义层整合入Sequential
    # net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    # Y = net(torch.rand(4, 8))
    # print(Y.mean())

    # 单独使用[带参数]自定义层
    linear = MyLinear(5,3)
    print(linear.weight)
    print(linear(torch.rand(2, 5)))
    #将[带参数]自定义层整合入Sequential
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))

