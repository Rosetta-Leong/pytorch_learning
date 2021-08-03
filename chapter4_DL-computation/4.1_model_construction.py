# -*-coding:utf-8-*-
# @File     :   2021/8/1 上午9:57
# @Author   :   Rosetta0
# @File     :   4.1_model_construction.py
# 学习如何通过继承nn.Module来自定义块，其中自定义块主要包含两部分
# (1)   块的结构：类成员
# (2)   前向传播：类成员函数

import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用`MLP`的父类`Block`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # 这里，`block`是`Module`子类的一个实例。我们把它保存在'Module'类的成员变量
            # `_children` 中。`block`的类型是OrderedDict。
            #必须指明为_modules属性
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

#具备常量参数与两个共享参数的全连接层的自定义块
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变。
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及`relu`和`dot`函数。
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


#嵌套使用
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))




if __name__ == "__main__":

    #pytorch
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    X = torch.rand(2, 20)
    print(net(X))

    #custom MLP
    net = MLP()
    print(net(X))

    #custom Sequential
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X))

    #custom block
    net = FixedHiddenMLP()
    print(net(X))

    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(chimera(X))





