# -*-coding:utf-8-*-
import torch


#Python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


if __name__ == "__main__":
    # 先看函数为标量的两个例子
    x = torch.arange(4.0, requires_grad=True)
    # 指明需要存储梯度,关于x的梯度存放于x.grad
    y = 2 * torch.dot(x, x)
    # 反向求导
    y.backward()
    # print(x.grad)
    # print(x.grad == 4 * x)

    # 默认情况下,Pytorch累积梯度，故需要清除以前的值
    x.grad.zero_()
    y = x.sum()
    y.backward()
    # print(x.grad)

    # 函数非标量(机器学习中少见)
    '''
    注意:这种情况都会转化为标量函数
    深度学习中，我们的目的不是计算微分矩阵，
    而是批量中每个样本单独计算的偏导数之和
    '''
    x.grad.zero_()
    y = x * x
    y.sum().backward()  # sum转化为标量函数
    # print(x.grad)

    #分离计算,将某些计算移动到记录的计算图之外
    '''
    我们可以分离 y 来返回一个新变量 u，
    该变量与 y 具有相同的值，但丢弃计算图中如何计算 y 的任何信息
    换句话说，梯度不会向后流经 u 到 x
    '''
    '''
    下面的反向传播函数计算 z = u * x 关于x的偏导数，
    同时将u作为常数处理，而不是 z = x * x * x 关于x的偏导数
    '''
    x.grad.zero_()
    y = x * x
    u = y.detach()  #貌似是把y转成常数
    #detach可用于固定神经网络中的参数
    z = u * x

    z.sum().backward()
    # print(x.grad == u)
    '''
    由于记录了 y 的计算结果，我们可以随后在 y 上调用反向传播，
    得到 y = x * x 关于的x的导数，这里是 2 * x
    '''
    x.grad.zero_()
    y.sum().backward()
    # print(x.grad == 2 * x)

    #Python控制流的梯度计算
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()
    print(a.grad == d/a)

