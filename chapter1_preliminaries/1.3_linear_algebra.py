# -*-coding:utf-8-*-
import torch

if __name__ == '__main__':

    A = torch.arange(20).reshape(5,-1)
    # print(A)
    # print(A.T)

    #对称矩阵
    B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
    # print(B)
    # print(B == B.T)

    X = torch.arange(24).reshape(2,3,4)
    # print(X)
    #注意结果输出，3和4代表最里层3*4矩阵，2代表最外层
    # tensor([[[ 0,  1,  2,  3],
    #      [ 4,  5,  6,  7],
    #      [ 8,  9, 10, 11]],
    #
    #     [[12, 13, 14, 15],
    #      [16, 17, 18, 19],
    #      [20, 21, 22, 23]]])

    #reshape并不改变所在地址，仅仅改变"view"视图
    #若想给reshape后的结果分配新地址示例如下
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
    # print(A)
    # print(B)
    # print(A.storage() == B.storage())

    #*运算：两个矩阵的按元素乘法称为 哈达玛积
    # print(A * B)
    a = 2
    #标量与张量乘除
    X = torch.arange(24).reshape(2, 3, 4)
    # print(a + X)
    # print((a * X).shape)

    '''
    指定张量沿哪一个轴来通过求和降低维度
    例如[5,4]沿轴0求和变为[4]一维向量
    沿轴1求和变为[5]一维向量
    '''
    # 例：求和所有行的元素来降维（轴0）
    A_sum_axis0 = A.sum(axis=0)
    # print(A)
    # print(A_sum_axis0)
    # print(A_sum_axis0.shape)
    #列降维（轴1）
    A_sum_axis1 = A.sum(axis=1)
    # print(A_sum_axis1)
    # print(A_sum_axis1.shape)
    #沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和
    #print(A.sum(axis=[0, 1]))   # Same as `A.sum()`
    #指定轴求均值
    # print(A.mean(axis=0))
    # print(A.sum(axis=0) / A.shape[0])

    #计算总和或均值时保持轴数不变
    #例如[5,4]对轴1求和且保持轴数不变则变为[5,1]
    sum_A = A.sum(axis=1, keepdims=True)
    # print(A)
    # print(sum_A)
    #由于 sum_A 在对每行进行求和后仍保持两个轴，可以通过广播将 A 除以 sum_A
    #广播机制必须保证维度相同，而m*n中m与n的具体值随意
    # print(A / sum_A)
    #cumsum:沿[某个轴计算 A 元素的累积总和]
    #此函数不会沿任何轴降低输入张量的维度
    # print(A.cumsum(axis=0))

    #两向量点积:    torch.dot(x,y)
    x = torch.arange(4,dtype=torch.float32)
    y = torch.ones(4,dtype=torch.float32)
    # print(torch.dot(x,y))
    #等价于    torch.sum(x * y)

    #矩阵-向量积：torch.mv(A,x)
    A = torch.arange(20).reshape(5,4)
    x = torch.arange(4)
    Ax = torch.mv(A,x)
    # print(A)
    # print(x)
    # print(Ax)

    #矩阵乘法：torch.mm(A, B)
    B = torch.ones(4, 3)
    # print(torch.mm(A, B))

    #torch.norm()   L2范数：平方和开根号
    #对向量来说norm即求L2范数
    u = torch.tensor([3.0, -4.0])
    # print(torch.norm(u))
    #L1范数：绝对值之和
    # print(torch.abs(u).sum())

    #对矩阵来说norm即求F范数，也是平方和开根号
    print(torch.norm(torch.ones((4, 9))))

