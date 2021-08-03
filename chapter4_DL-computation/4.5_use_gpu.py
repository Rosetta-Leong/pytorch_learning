# -*-coding:utf-8-*-
# @File     :   2021/8/1 下午4:15
# @Author   :   Rosetta0
# @File     :   4.5_use_gpu.py

#注意以下 [部分] 代码需要多张GPU以完成，但colab也给不了这么多(反正只给了我一张卡)
#链接：https://d2l.ai/chapter_deep-learning-computation/use-gpu.html#computing-devices
#标题右边--colab链接

import torch
from torch import nn

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

if __name__ == "__main__":

    # CPU用torch.device('cpu')表示
    # 使用torch.cuda.device(f'cuda:{i}')来表示第 𝑖 块GPU（ 𝑖 从0开始）
    # print(torch.device('cpu'))
    # print(torch.cuda.device('cuda'))
    # print(torch.cuda.device('cuda:1'))
    # print(torch.cuda.device_count())
    #
    # print(try_gpu())
    # print(try_gpu(10))
    # print(try_all_gpus())

    #查询张量所在设备
    # x = torch.tensor([1, 2, 3])
    # print(x.device) #默认为CPU


#################不同GPU间Tensor创建传输###################
    #创建Tensor时指定Device
    X = torch.ones(2, 3, device=try_gpu())
    # print(X)
    # #选第二块GPU（我没有，所以未运行）
    # Y = torch.rand(2, 3, device=try_gpu(1))
    # print(Y)
    #
    # #为执行不同device上的X+Y，需进行复制
    # Z = X.cuda(1)   #在cuda:0上的X --> cuda:1上的Z
    # print(X)
    # print(Z)
    # print(Y + Z)
    # print(Z.cuda(1) is Z)   #已在该device上的Tensor不会自己拷贝自己
################################################################

    net = nn.Sequential(nn.Linear(3, 1))    #创建仍在CPU上
    net = net.to(device=try_gpu())  #.to()方法挪到GPU上
    print(net(X))   #X也在0号GPU上
    print(net[0].weight.data.device)    #确认模型参数存储在同一个GPU上