# -*-coding:utf-8-*-
# @File     :   2021/8/10 下午8:29
# @Author   :   Rosetta0
# @File     :   6.7_Resnet.py

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 实现残差块
class Residual(nn.Module):
    
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)   #在原张量基础上修改，不生成新对象

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 实现ResNet模块，每个模块由多个残差块组成
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    #每个模块使用若干个同样输出通道数的残差块
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
            #除第一个模块外，每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
            #第一个模块的通道数同输入通道数一致，由于之前已经使用了步幅为 2 的最大池化层，所以无须减小高和宽
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


if __name__ == "__main__":

    # 输入输出形状一致
    # blk = Residual(3, 3)
    # X = torch.rand(4, 3, 6, 6)
    # Y = blk(X)
    # print(Y.shape)

    #通道数加倍，高宽减半
    # blk =   Residual(3, 6, use_1x1conv=True, strides=2)
    # print(blk(X).shape)

    '''
    ResNet-18
    前两层跟之前介绍的 GoogLeNet 中的一样： 
    在输出通道数为64、步幅为2的 7×7 卷积层后，接步幅为2的 3×3 的最大池化层
    不同之处在于 ResNet 每个卷积层后增加了批量归一化层。
    '''
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 接着在 ResNet 加入所有残差块，这里每个模块使用 2 个残差块
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    #与 GoogLeNet 一样，在 ResNet 中加入全局平均池化层，以及全连接层输出
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

    # X = torch.rand(size=(1, 1, 224, 224))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    #loss 0.013, train acc 0.996, test acc 0.923
    #1155.8 examples/sec on cuda:0
    # ResNet NB!!!