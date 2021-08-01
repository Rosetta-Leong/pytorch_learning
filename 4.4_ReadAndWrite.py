# -*-coding:utf-8-*-
# @File     :   2021/8/1 下午3:03
# @Author   :   Rosetta0
# @File     :   4.4_ReadAndWrite.py

import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

if __name__ == "__main__":
    #存取Tensor
    x = torch.arange(4)
    # torch.save(x, 'x-file')
    # x2 = torch.load('x-file')
    # print(x2)

    #存取Tensor List
    y = torch.zeros(4)
    # torch.save([x, y], 'x-files')
    # x2, y2 = torch.load('x-files')
    # print((x2, y2))

    #存取
    # mydict = {'x':x, 'y':y}
    # torch.save(mydict, 'mydict')
    # mydict2 = torch.load('mydict')
    # print(mydict2)

    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    #存模型参数
    torch.save(net.state_dict(), 'mlp.params')

    #取模型参数-->NEW MLP
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    # print(clone.eval())

    #JUDGE the latter and the former MLP's params
    Y_CLONE = clone(X)
    print(Y_CLONE == Y)