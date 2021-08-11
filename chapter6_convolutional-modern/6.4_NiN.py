# -*-coding:utf-8-*-
# @File     :   2021/8/8 下午4:58
# @Author   :   Rosetta0
# @File     :   6.4_NiN.py

import torch
from torch import nn
from d2l import torch as d2l

#通过1 * 1卷积，相当于对每个像素做全连接操作
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

if __name__ == "__main__":

    #结构类似于AlexNet（卷积的各项参数）, 仅把卷积层替换为NiN块
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(p=0.5),
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        # 全局平均池化，得到高宽均为1，即输出 (批量大小 * 10 * 1 * 1)
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten()    # batch_size * 10
    )

    # x = torch.rand(size=(1, 1, 224, 224))
    # for layer in net:
    #     x = layer(x)
    #     print(layer.__class__.__name__, 'output size:\t', x.shape)


    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # loss 0.336, train acc 0.876, test acc 0.879
    # 973.5 examples/sec on cuda:0