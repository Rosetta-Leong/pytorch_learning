# -*-coding:utf-8-*-
# @File     :   2021/8/8 上午10:50
# @Author   :   Rosetta0
# @File     :   6.2_AlexNet.py

import torch
from torch import nn
from d2l import torch as d2l



if __name__ == "__main__":

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    # 使用更小的学习速率训练，这是因为网络更深更广、图像分辨率更高，训练卷积神经网络就更昂贵

    net = nn.Sequential(
        # 使用Fashion-MNIST数据集，输入为单通道

        # 这里为什么要padding，搞不懂啊？
        # CS231n里面提到AlexNet是修改了input图像的尺寸，从224 * 224改为227 * 227，否则原论文结构（pytorch padding=2）不work
        # https: // www.youtube.com / watch?v = LxfUGhug - iQ & t = 10s（快进到46: 55）
        # 而这里沐神的应对方式是把第一个卷积层的padding设置为1
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),    # 进全连接层前要展平！！
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10) # Fashion-MNIST输出类别为10
    )

    # 如下代码可知经过展平后的tensor形状
    # x = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
    # for layer in net:
    #     x = layer(x)
    #     print(layer.__class__.__name__, 'Output shape:\t', X.shape)

    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    #结果：loss 0.334, train acc 0.878, test acc 0.882
