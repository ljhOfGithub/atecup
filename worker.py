import torch
import torch.nn.functional as F
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from preprocess import CompDataset
from preprocess import get_user_data


class Worker(object):
    def __init__(self, user_idx):#
        self.user_idx = user_idx #调用worker的用户索引
        self.data, self.edges = get_user_data(self.user_idx)  # The worker can only access its own data 获取数据和边
        self.ps_info = {}

    def preprocess_worker_data(self): #用户的对应worker预处理原本的csv数据
        self.data = self.data[self.data['class'] != 2] #data指workerdata.csv中的数据，非第二类的数据
        x = self.data.iloc[:, 2:] #第二列及之后的列的所有行，排除txid列和class列
        x = x.reset_index(drop=True) #建立索引列，由序号组成，从0开始
        x = x.to_numpy().astype(np.float32) #改成numpy数组
        y = self.data['class'] #类那一列
        y = y.reset_index(drop=True) #drop：当指定drop=False时，则索引列会被还原为普通列；否则，经设置后的新索引值被会丢弃。默认为False。
        x[x == np.inf] = 1. #浮点数
        x[np.isnan(x)] = 0. #在使用numpy数组的过程中时常会出现nan或者inf的元素，可能会造成数值计算时的一些错误，使nan和inf能够最简单地转换成相应的数值。
        self.data = (x, y) #重新组合csv的数据，分为x和y对象

    def round_data(self, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round. 在某轮产生对应用户的数据

        Args:
            n_round: int, round number 回合数
            n_round_samples: int, the number of samples this round 该回合中的例子数
        """

        if n_round_samples == -1: #该回合没有例子
            return self.data

        n_samples = len(self.data[1]) #原csv的行数
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))  #打乱原来的有序数组（0，1，2...），返回一个随机排列的数组
#从数组中随机抽取元素
#numpy . random. choice(a, size=None, replace=True, p=None )
#从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的 数组
#replace: True表示可以取相同数字，False表 示不可以取相同数字
#数组p:与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

        return self.data[0][choices], self.data[1][choices] #获取节点的数据，返回一个随机的数组，第一行随机排序后的列，第二行随机的列

    def receive_server_info(self, info):  # receive info from PS 从参数服务器中接收ps的信息（训练后的准确率）
        self.ps_info = info

    def process_mean_round_train_acc(self):  # process the "mean_round_train_acc" info from server 从服务器中接收并处理mean_round_train_acc信息
        mean_round_train_acc = self.ps_info["mean_round_train_acc"]
        # You can go on to do more processing if needed

    def user_round_train(self, model, device, n_round, batch_size, n_round_samples=-1, debug=False):#用户回合的训练
        X, Y = self.round_data(n_round, n_round_samples) #
        data = CompDataset(X=X, Y=Y) #载入数据
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
        )#训练数据加载器

        model.train() #训练

        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model = model.to(device)
        for batch_idx, (data, target) in enumerate(train_loader): #将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            data, target = data.to(device), target.to(device) #将从服务器取出的数据放入神经网络进行训练
            # import ipdb 一款集成了Ipython的Python代码命令行调试工具
            # ipdb.set_trace() 在**ipdb.set_trace()**的行会停下来，进入交互式调试模式
            # print(data.shape, target.shape)
            output = model(data) #构造神经网络对象
            loss = F.nll_loss(output, target) #negative log likelihood loss 计算误差
            total_loss += loss #加和误差
            loss.backward() #求导
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability 得到最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item() #view_as返回被视作与给定的tensor（参数）相同大小的调用该函数的tensor
            #eq判断对象是否相同，sum()对给定对象求和，item()返回可遍历的(键, 值) 元组数组
            prediction.extend(pred.reshape(-1).tolist()) #添加到预测数据列表中
            real.extend(target.reshape(-1).tolist()) #添加到实际数据列表中

        grads = {'n_samples': data.shape[0], 'named_grads': {}} #梯度字典，data.shape[0]行数，共4000+
        for name, param in model.named_parameters(): #返回各层中参数名称和数据
            grads['named_grads'][name] = param.grad.detach().cpu().numpy() #detach阻断反向传播，numpy()将tensor转换为numpy数组，cpu()将变量放在cpu上，
            # 将param的梯度存储到字典中

        worker_info = {}
        worker_info["train_acc"] = correct / len(train_loader.dataset) #存储训练的准确率

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(train_loader.dataset)))

        return grads, worker_info
