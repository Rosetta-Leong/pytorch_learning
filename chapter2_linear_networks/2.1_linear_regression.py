# -*-coding:utf-8-*-
import random
import torch
from torch.utils import data
from torch import nn #nn:神经网络缩写
import numpy as np
from d2l import torch as d2l

#构造人造数据集
def synthetic_data(w,b,num_examples):
    """生成y = Xw + b + 噪声"""
    '''
    使用线性模型参数 𝐰=[2,−3.4]⊤ 、 𝑏=4.2 
    和噪声项 𝜖 生成数据集及其标签：
    '''
    X = torch.normal(0,1,(num_examples, len(w)))    #num_examples * 2的矩阵
    y = torch.matmul(X,w) + b   #num_examples * 1的矩阵
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

#函数功能：数据集读取
#输入：批量大小batch_size, 特征矩阵features， 标签向量labels
#输出：大小为batch_size的小批量样本（特征+标签）
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) #range相当于从0到n-1这些数，然后转成List格式
    random.shuffle(indices) #将索引打乱，强啊！
    #例如batch_size为16的话，相当于16个为一组
    # i每轮循环步进16次，即每轮循环i为当前新batch的首项
    for i in range(0,num_examples,batch_size):
        #每轮循环中从list中切片[i:i+16]，从而得到当前batch(一组16个数据)
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        #yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开

#定义线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

#定义损失函数(均方误差）
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

#定义优化算法:小批量随机梯度下降
#params为模型参数w与b，lr为学习率Learning rate
def sgd(params, lr, batch_size):
    #更新时无需计算梯度
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def my_train():
    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练参数设置
    lr = 0.03  # default:0.03
    num_epochs = 3  # 整个数据扫三遍
    net = linreg  # 如此可以方便后期换成不同的模型
    loss = squared_loss
    batch_size = 10  # default:10

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 由小批量计算损失
            # 因为`l`形状是(`batch_size`, 1)，而不是一个标量
            # `l`中的所有元素被加到一起，并以此计算关于[`w`, `b`]的梯度
            # 由此优化时需除上一个 batch_size
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')



#以下为使用PyTorch实现：

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def PyTorch_train():

    batch_size = 10
    #数据读取
    PyTorch_data_iter = load_array((features, labels), batch_size)

    #网络结构定义:单层神经网络：仅一层全连接层
    # nn.Linear 中第一个指定输入特征形状，即 2，
    # 第二个指定输出特征形状，输出特征形状为单个标量，因此为 1
    net = nn.Sequential(nn.Linear(2,1))


    #模型参数定义
    #通过_ 结尾的方法将参数替换，从而初始化参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    #损失函数
    '''平方𝐿2范数,默认情况下，它返回[所有样本]损失的平均值（mean）
    reduction参数的不同取值
    none: no reduction will be applied.
    mean: the sum of the output will be divided by the number of elements in the output.
    sum: the output will be summed    
    若reduction = 'sum',则在优化时应除上样本数,本例中每次传入10个样本（batch_size）,相当于lr = lr/10
    '''
    loss = nn.MSELoss()

    #实例化SGD
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    #训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in PyTorch_data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward() #与从零实现相比，均值计算已在Loss函数中完成
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)


if __name__ == "__main__":

    #数据集生成有关参数
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w,true_b,1000)
    '''
    features 中的每一行都包含一个二维数据样本，
    labels 中的每一行都包含一维标签值（一个标量)
    '''
    # print('features:', features[0], '\nlabel:', labels[0])
    '''
    通过生成第二个特征 features[:, 1] 和 labels 的散点图
    可以直观地观察到两者之间的线性关系。
    '''
    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, (1)].detach().numpy(),
    #                 labels.detach().numpy(), 1);

    #从零开始实现
    my_train()

    #torch实现
    PyTorch_train()

    #Q:
    # 1.似乎epoch相同的情况下，从零开始实现效果要好些？

    # 2.loss function reduction= 'mean' , lr=0.03:
    # epoch 1, loss 0.000361
    # epoch 2, loss 0.000095
    # epoch 3, loss 0.000095
    # w的估计误差： tensor([-0.0002, -0.0002])
    # b的估计误差： tensor([8.6784e-05])
    #
    # loss function reduction = 'sum', lr =0.03/batch_size=0.003
    # epoch 1, loss 0.176310
    # epoch 2, loss 0.091707
    # epoch 3, loss 0.092358
    # w的估计误差： tensor([0.0007, 0.0012])
    # b的估计误差： tensor([0.0005])
    #这两种方式理论上等价，为何loss function取mean显著由于取sum的方式