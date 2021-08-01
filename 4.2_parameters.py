# -*-coding:utf-8-*-
# @File     :   2021/8/1 上午10:24
# @Author   :   Rosetta0
# @File     :   4.2_parameters.py

import torch
from torch import nn

#生成嵌套Sequential
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4),
                         nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

#不同的参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        # 下划线表明该方法是一个替换函数，不返回值，而是把传入的某个参数进行替换
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

#自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

if __name__ == "__main__":

    X = torch.rand(size=(2, 4))
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    # print(net[2].state_dict())
    # print(type(net[2].bias))
    # print(net[2].bias)
    # print(net[2].bias.data) #.data访问参数本身值
    # print(net[2].weight.grad == None)   #grad访问参数对应的梯度

    # 访问所有参数
    # print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    # print(*[(name, param.shape) for name, param in net.named_parameters()])
    # print(net.state_dict()['2.bias'].data)

    #嵌套块收集参数
    # rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    # print(rgnet)
    # print(rgnet[0][1][0].bias.data)

    # net.apply(init_normal)  #使用该方法自定义参数初始化时，自定义函数名后不加括号
    # net.apply(init_constant)
    # print(net[0].weight.data[0], net[0].bias.data[0])

    #对不同块实行不同的初始化方法
    # net[0].apply(xavier)
    # net[2].apply(init_42)
    # print(net[0].weight.data[0])
    # print(net[2].weight.data)

    #自定义初始函数
    # net.apply(my_init)
    # print(net[0].weight[:2])

    #直接手撸
    # net[0].weight.data[:] += 1
    # net[0].weight.data[0, 0] = 42
    # print(net[0].weight.data[0])


##################参数共享(绑定)######################################
    # 我们需要给共享层一个名称，以便可以引用它的参数。
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                        nn.ReLU(), nn.Linear(8, 1))
    # 检查参数是否相同
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    # 确保它们实际上是同一个对象，而不只是有相同的值。
    print(net[2].weight.data[0] == net[4].weight.data[0])
################################################################