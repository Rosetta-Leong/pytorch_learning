# -*-coding:utf-8-*-
# @File     :   2021/7/23 下午6:24
# @Author   :   Rosetta0
# @File     :   2.3_softmax

import torch
from IPython import display
from d2l import torch as d2l
from torch import nn

#softmax中的分母(归一化常数)即热统中的配分函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  #对轴1求和，即对[行]求和
    return X_exp/partition #这里应用了广播机制

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

'''
:param y_hat: 预测值 
:param y: 真实值（标签）
:return: 交叉熵损失函数 
'''
def cross_entropy(y_hat, y):
    return -torch.log( y_hat[ range(len(y_hat)),  y] )
    #此处使用索引机制，巧妙提取出每个样本真实值（标签）对应的预测概率

#计算预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] >1:
        y_hat = y_hat.argmax(axis=1) #沿列查找最大值下标，即获得预测类别
    cmp = y_hat.type(y.dtype) == y
    return  float(cmp.type(y.dtype).sum())

#计算在指定数据集上的模型精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric = Accumulator(2) #正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] /metric[1]

#n变量累加器实现
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#训练过程[仅迭代一次]
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #分别为训练损失总和、训练准确度总和、样本数
    metric =Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  #torch版本优化器和损失函数
            updater.zero_grad()
            l.backward()    #此处loss function一般为mean方式
            updater.step
            metric.add(float(l) * len(y), accuracy(y_hat,y), y.size().numel())
        else:   #自定义版本
            l.sum().backward() #自定义的损失函数值是一个向量，所以对其求和
            updater(X.shape[0]) #根据批量大小来update一下
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    #返回训练损失和准确率
    return metric[0] / metric[2], metric[1]/metric[2]

#以下为画图相关
class Animator:
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

def add(self, x, y):
    # 向图表中添加多个数据点
    if not hasattr(y, "__len__"):
        y = [y]
    n = len(y)
    if not hasattr(x, "__len__"):
        x = [x] * n
    if not self.X:
        self.X = [[] for _ in range(n)]
    if not self.Y:
        self.Y = [[] for _ in range(n)]
    for i, (a, b) in enumerate(zip(x, y)):
        if a is not None and b is not None:
            self.X[i].append(a)
            self.Y[i].append(b)
    self.axes[0].cla()
    for x, y, fmt in zip(self.X, self.Y, self.fmts):
        self.axes[0].plot(x, y, fmt)
    self.config_axes()
    display.display(self.fig)
    display.clear_output(wait=True)

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


#训练函数（多epoch）
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

#############分割线##############
#######以下为Pytorch实现softamx回归有关函数########

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) #均值默认为0，标准差为0.01


if __name__ == "__main__":

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    lr = 0.1
    num_epochs = 10

    #以下为从零实现

    # 原始数据集中的每个样本都是  28×28  的图像
    # 展平每个图像，把它们看作长度为784的向量[1 * 784]行向量
    num_inputs = 784
    # 数据集有十个类别，故网络输出维度为10
    num_outputs = 10
    # 权重[784 * 10], 偏置[1 * 10]
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # predict_ch3(net, test_iter)
    #提示：实时反映train acc 和 test acc的图加载不出来, 估计训练函数写错了，因为pytorch版的没问题
    #     但是测试图片无误


    #Pytorch实现

    #Pytorch不会隐式地调整输入的形状
    #故定义展平层(FLatten)在线性层前调整网络输入的形状
    torchnet = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    torchnet.apply(init_weights)
    torchloss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(torchnet.parameters(), lr=0.1)
    d2l.train_ch3(torchnet, train_iter, test_iter, torchloss, num_epochs, trainer)
    predict_ch3(torchnet, test_iter)
