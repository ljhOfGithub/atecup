from datetime import datetime
import os
import shutil
import unittest
import pickle

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from context import FederatedSGD
from context import PytorchModel
from learning_model import FLModel
from preprocess import get_test_data


class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir, resultdir):
        self.round = 0 #回合数
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.worker_info = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedSGD(
            model=PytorchModel(torch=torch,
                               model_class=FLModel,
                               init_model_path=self.init_model_path,
                               optim_name='Adam'), #使用adam优化器
            framework='pytorch',
        ) #
        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        if not os.path.exists(self.testworkdir): #测试文件夹
            os.makedirs(self.testworkdir)
        if not os.path.exists(self.RESULT_DIR): #结果文件夹
            os.makedirs(self.RESULT_DIR)

        self.test_data, self.test_edges = get_test_data() #测试的节点和边
        self.preprocess_test_data() #预处理

        self.round_train_acc = [] #初始化该轮的训练精确度

    def preprocess_test_data(self): #预处理
        self.predict_data = self.test_data[self.test_data['class'] == 3]  # to be predicted 指定某一类的数据
        self.predict_data_txId = self.predict_data[['txId', 'Timestep']] #获取指定数据的txId值
        x = self.predict_data.iloc[:, 3:] #iloc函数：取第三列及其之后列的所有行
        x = x.reset_index(drop=True) #
        x = x.to_numpy().astype(np.float32) #np.float32的numpy数组
        x[x == np.inf] = 1. #无穷大
        x[np.isnan(x)] = 0. #判断是否是空值
        self.predict_data = x #预测数据

    def get_latest_model(self): #获取最新一轮的模型路径
        if not self.rounds_model_path: #如果没有rounds_model_path则返回初始化模型路径
            return self.init_model_path

        if self.round in self.rounds_model_path: #如果是指定的回合
            return self.rounds_model_path[self.round] #返回对应回合的模型路径

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads):  # receive grads info from worker 获取梯度信息
        self.current_round_grads.append(grads)

    def receive_worker_info(self, info):  # receive worker info from worker 获取工人信息
        self.worker_info = info

    def process_round_train_acc(self):  # process the "round_train_acc" info from worker 处理round_train_acc
        self.round_train_acc.append(self.worker_info["train_acc"])

    def print_round_train_acc(self): #打印round_train_acc
        mean_round_train_acc = np.mean(self.round_train_acc) * 100
        print("\nMean_round_train_acc: ", "%.2f%%" % (mean_round_train_acc))
        self.round_train_acc = []
        return {"mean_round_train_acc": mean_round_train_acc
                }

    def aggregate(self): #汇总
        self.aggr(self.current_round_grads) #

        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1 #汇总完回合数+1
        self.current_round_grads = []

        return info

    def save_prediction(self, predition): #prediction预测

        predition.to_csv(os.path.join(self.RESULT_DIR, 'result.csv'), index=0) #存储为csv数据

    def save_model(self, model): #存储模型

        with open(os.path.join(self.RESULT_DIR, 'model.pkl'), 'wb') as fout:
            pickle.dump(model, fout) #序列化数据

    def save_testdata_prediction(self, model, device, test_batch_size): #存储测试数据的预测
        self.test_data
        loader = torch.utils.data.DataLoader( #数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器，作用就是实现数据以什么方式输入到什么网络中
            self.predict_data, #自定义的Dataset
            batch_size=test_batch_size, #
            shuffle=False, #在每个epoch中对整个数据集data进行shuffle重排，默认为False
        )
        prediction = [] #预测
        with torch.no_grad(): #用来禁止梯度的计算
            for data in loader: #取出神经网络的数据进行存储
                pred = model(data.to(device)).argmax(dim=1, keepdim=True) #argmax返回跨维度的张量最大值的索引,keepdim输出张量是否保持dim
                prediction.extend(pred.reshape(-1).tolist()) #转为一维数组
        self.predict_data_txId['prediction'] = prediction #

        self.save_prediction(self.predict_data_txId) #存储预测数据
        self.save_model(model) #存储模型

#集群中的节点可以分为计算节点和参数服务节点两种。其中，计算节点负责对分配到自己本地的训练数据（块）计算学习，并更新对应的参数；
# 参数服务节点采用分布式存储的方式，各自存储全局参数的一部分，并作为服务方接受计算节点的参数查询和更新请求。

