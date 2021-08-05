# -*-coding:utf-8-*-
# @File     :   2021/8/5 下午4:47
# @Author   :   Rosetta0
# @File     :   5.5_LeNet.py

import torch
from torch import nn
from d2l import torch as d2l

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

#为保证在可在GPU上实现，对如下函数修改
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""

    #若未指定device，则查询网络参数存于何device并以此为准
    if isinstance(net, nn.Module):
        net.eval()  #模型处于predict阶段
        if not device:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)

    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]

#训练函数更改
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU"""

    #在函数内部定义的函数要在函数内部调用
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('training on:', device)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        #训练损失之和， 训练准确率之和， 范例数
        metric = d2l.Accumulator(3)
        net.train() #模型设置为train模式

        #enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
        #同时列出数据和数据下标，一般用在for循环中
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add((l * X.shape[0]), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == "__main__":

    net = nn.Sequential(Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.Sigmoid(),
                        nn.Linear(84,10)
                        )
    #简单测试一下各层输出
    # X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)

    #Fashion-MNIST表现
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

