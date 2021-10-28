import torch.nn as nn
import torch.nn.functional as F


class FLModel(nn.Module): #继承module，定义自己的网络，输入神经网络进行学习然后归一化,神经网络设计的模块化接口
    def __init__(self): #具有可学习参数的层
        super().__init__()
        self.fc1 = nn.Linear(165, 50) # 输入/输出的二维张量的大小
        self.fc5 = nn.Linear(50, 2) # 实例化了两个nn.Linear层，并将它们作为成员变量

    def forward(self, x): #backward函数自动实现，在使用pytorch的时候，模型训练时，不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
        x = self.fc1(x) #第一层
        x = F.relu(x) # 线性整流函数
        x = self.fc5(x) #第五层
        output = F.log_softmax(x, dim=1) #归一化指数函数后取对数，当dim=1时， 是对某一维度的列进行softmax运算

        return output
