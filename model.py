from datetime import datetime
import os
import shutil
import unittest

import numpy as np
from sklearn.metrics import classification_report # classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
import torch
import torch.nn.functional as F

from context import FederatedSGD
from context import PytorchModel
from learning_model import FLModel
from worker import Worker
from server import ParameterServer
#server 节点的主要功能是保存模型参数、接受 worker 节点计算出的局部梯度、汇总计算全局梯度，并更新模型参数
#worker 节点的主要功能是保存各部分训练数据，从 server 节点拉取最新的模型参数，根据训练数据计算局部梯度，上传给 server 节点。

class FedSGDTestSuit(unittest.TestCase): #测试类
    RESULT_DIR = 'result' #创建结果文件夹
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/' #测试集路径

    def setUp(self):
        self.seed = 0
        self.use_cuda = False #
        self.batch_size = 64 #一次训练所抓取的数据样本数量
        self.test_batch_size = 1000 #一次测试所抓取的数据样本数量
        self.lr = 0.001 #学习率
        self.n_max_rounds = 100 #最大训练回合数
        self.log_interval = 20 #
        self.n_round_samples = 1600 #一轮的例子数
        self.testbase = self.TEST_BASE_DIR #测试的基文件夹
        self.n_users = 40 #用户数（用于分布式机器数）
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir) #创建测试文件夹

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md') #初始化模型
        torch.manual_seed(self.seed) #用于生成随机数种子

        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path) #存储模型
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir, resultdir=self.RESULT_DIR) #ParameterServer对象

        self.workers = []
        for u in range(0, self.n_users):
            self.workers.append(Worker(user_idx=u)) #根据用户数添加worker

    def _clear(self):
        shutil.rmtree(self.testworkdir) #递归删除文件夹下的所有子文件夹和子文件

    def tearDown(self): #清除字典中所有的项
        self._clear()

    def test_federated_SGD(self): #联合的局部随机梯度下降（SGD）
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu") #torch.device代表将torch.Tensor分配到的设备，根据默认是使用cpu

        # let workers preprocess data
        for u in range(0, self.n_users):  #n_users用户数量
            self.workers[u].preprocess_worker_data() #对应的worker预处理对应的数据

        training_start = datetime.now() #
        model = None
        for r in range(1, self.n_max_rounds + 1): #训练n_max_rounds轮
            path = self.ps.get_latest_model() #
            start = datetime.now() #
            for u in range(0, self.n_users):
                model = FLModel() #神经网络模型
                model.load_state_dict(torch.load(path)) #load_state_dict：使用反序列化的状态字典加载模型的参数字典
                model = model.to(device) #返回具有特定设备类型和(可选)特定数据类型的神经网络
                grads, worker_info = self.workers[u].user_round_train(model=model, device=device, n_round=r,
                                                                      batch_size=self.batch_size,
                                                                      n_round_samples=self.n_round_samples) #某一个用户的训练数据的准确率

                self.ps.receive_grads_info(grads=grads) #
                self.ps.receive_worker_info(
                    worker_info)  # The transfer of information from the worker to the server requires a call to the "ps.receive_worker_info"
                self.ps.process_round_train_acc() #

            self.ps.aggregate() #
            print('\nRound {} cost: {}, total training cost: {}'.format(
                r,
                datetime.now() - start, #计算花费时间
                datetime.now() - training_start, #训练时间
            ))

            if model is not None and r % self.log_interval == 0:
                server_info = self.ps.print_round_train_acc()  # print average train acc and return training accuracy训练精确度
                for u in range(0, self.n_users):  # transport average train acc to each worker
                    self.workers[u].receive_server_info(
                        server_info)  # The transfer of information from the server to the worker requires a call to the "worker.receive_server_info" 从server到worker的信息传递需要调用函数worker.receive_server_info
                    self.workers[u].process_mean_round_train_acc()  # workers do processing 处理该回合的训练数据

                self.ps.save_testdata_prediction(model=model, device=device, test_batch_size=self.test_batch_size) #存储测试数据的标识

        if model is not None:
            self.ps.save_testdata_prediction(model=model, device=device, test_batch_size=self.test_batch_size) #存储测试数据的标识


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedSGDTestSuit('test_federated_SGD'))
    return suite


def main():
    runner = unittest.TextTestRunner() #文本类测试用例运行器
    runner.run(suite()) #执行测试用例或测试集


if __name__ == '__main__':
    main()

