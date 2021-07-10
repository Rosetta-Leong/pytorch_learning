import torch


if __name__ == '__main__':

    x = torch.arange(12)    #创建行向量，元素为浮点数
    # print(x)
    #
    # print(x.shape)  #shape属性访问张量形状
    # print(x.numel())    #numel()访问元素总数--括号！！！

    X = x.reshape(3,4)  #reshape改形状
    #宽高任给其一即可，-1代表张量自动确定
    #X = x.reshape(3,-1)
    #X = x.reshape(-1,4)
    #print(X)

    #全0张量
    y = torch.zeros((2,3,4))
    #print(y)

    #全1张量
    z = torch.ones((2,3,4))
    #print(z)

    #randn：每个元素都从均值为0、标准差为1的标准高斯（正态）分布中随机采样
    g = torch.randn(3,4)
    #print(g)

    #通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
    #最外层的列表对应于轴 0，内层的列表对应于轴 1。
    h = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    #print(h)

    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)   #求幂


