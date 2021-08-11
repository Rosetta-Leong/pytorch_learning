# -*-coding:utf-8-*-
# @File     :   2021/8/10 下午6:30
# @Author   :   Rosetta0
# @File     :   6.6_BatchNorm.py

import torch
from torch import nn
from d2l import torch as d2l

# moving_mean, moving_var分别为全局均值、方差（不是小批量上的）
# eps为方差后面加上的小项，避免方差为0即归一化时除0的情况
# momentum用于更新moving_mean和moving_var
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        #推理阶段一般使用整个数据集上的均值和方差
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)   #2D：全连接层， 4D：卷积层
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            #在每个通道维度上对其他维度求平均，得到1*n*1*1
            #需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差,不断更新可以逼近真实意义上的全局均值、方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


'''
[创建一个正确的 BatchNorm 图层]
1.在训练过程中更新拉伸 gamma 和偏移 beta
2.保存均值和方差的移动平均值，以便在模型预测期间使用
'''
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        #python3 直接写成 ： super().__init__()
        #python2 必须写成 ：super(本类名,self).__init__()
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果 `X` 不在内存上，将 `moving_mean` 和 `moving_var`
        # 复制到 `X` 所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的 `moving_mean` 和 `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)
        return Y





if __name__ == "__main__":

    # 使用批量归一化层的 LeNet
    # net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4),
    #                     nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
    #                     nn.Conv2d(6, 16,
    #                               kernel_size=5), BatchNorm(16, num_dims=4),
    #                     nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
    #                     nn.Flatten(), nn.Linear(16 * 4 * 4, 120),
    #                     BatchNorm(120, num_dims=2), nn.Sigmoid(),
    #                     nn.Linear(120, 84), BatchNorm(84, num_dims=2),
    #                     nn.Sigmoid(), nn.Linear(84, 10))

    #pytorch实现
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6),
                        nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16,
                                  kernel_size=5), nn.BatchNorm2d(16),
                        nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Flatten(), nn.Linear(16 * 4 * 4, 120),
                        nn.BatchNorm1d(120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.BatchNorm1d(84),
                        nn.Sigmoid(), nn.Linear(84, 10))

    #LeNet + BN
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    '''
    loss 0.244, train acc 0.911, test acc 0.861
    43140.9 examples/sec on cuda:0
    从零实现过拟合有点严重???
    '''
    # print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
    #pytorch实现：
    '''
    loss 0.250, train acc 0.909, test acc 0.851
    59048.3 examples/sec on cuda:0
    '''