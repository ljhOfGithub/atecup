import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

TRAINDATA_DIR = 'TrainData/TrainData/'
TESTDATA_PATH = './Test_X/Test_X.pkl'

class CompDataset(object): #比赛数据集
    def __init__(self, X, Y): #
        self.X = X
        self.Y = Y

        self._data = [(x, y) for x, y in zip(X, Y)] #打包X，Y，组成元组，zip是元组的列表

    def __getitem__(self, idx):#getter
        return self._data[idx] #返回对应坐标的元组

    def __len__(self):#长度（维数）
        return len(self._data) #返回元组列表的长度



def get_user_data(user_idx): #用于提取csv中用于训练的worker节点数据和交易流数据
    train_data_path  = TRAINDATA_DIR+'WorkerData_{}.csv'.format(user_idx)
    t_data = pd.read_csv(train_data_path) #读取训练数据的交易数据
    # read edges data
    train_edges_path = TRAINDATA_DIR+'WorkerDataEdges_{}.csv'.format(user_idx) # 图中节点是每笔交易，边代表的是交易时双方之间存在的交易流。
    t_edges = pd.read_csv(train_edges_path) #读取训练数据的交易流数据
    return t_data,t_edges #返回交易数据和交易流数据


def get_test_data(): #提取测试数据
    with open(TESTDATA_PATH, 'rb') as fin: #打开测试文件 binary format for reading
        data = pickle.load(fin) # 反序列化数据流
    return data["data"],data["edges"] #返回节点数据和边的交易数据