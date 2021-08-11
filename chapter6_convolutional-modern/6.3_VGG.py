# -*-coding:utf-8-*-
# @File     :   2021/8/8 下午12:49
# @Author   :   Rosetta0
# @File     :   6.3_VGG.py

import torch
from torch import nn
from d2l import torch as d2l

# 实现VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels  #输出通道数都一样，第一次进来时输入通道可能不同
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 实现VGG网络
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1 # Fahsion-MNIST 单通道
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels # 保证本层输出与下一层输入的通道一致！！
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         # 全连接层部分
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(4096, 10)
    )



if __name__ == "__main__":

    conv_arch = ((1,64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)

    # X = torch.randn(size=(1, 1, 224, 224))
    # for blk in net:
    #     X = blk(X)
    #     print(blk.__class__.__name__, 'output shape:\t', X.shape)

    ratio = 4   #所有通道数/4以减小模型大小，便于训练测试
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    '''
    loss 0.173, train acc 0.937, test acc 0.917
    606.9 examples/sec on cuda:0
    '''