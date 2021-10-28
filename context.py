from abc import ABC
from abc import abstractmethod
import os

import numpy as np
import torch

from aggretator import aggregate_grads


def random_str(n):
    return hex(int.from_bytes(os.urandom(n), byteorder='big'))[2:]  # 十六进制随机字符串



class ModelBase(ABC):  # 继承抽象基类abc
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])  # 设置self的变量k的值为kwargs[k]

    @abstractmethod
    def update_grads(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

#本地变量： torch
class PytorchModel(ModelBase):  # 传入抽象类
    def __init__(self,
                 torch,
                 model_class, #learning_model对象
                 init_model_path: str = '',
                 lr: float = 0.01,
                 optim_name: str = 'Adam',
                 cuda: bool = False):
        """Pytorch 封装.

        参数：
            torch: torch 库
            model_class: 训练模型类
            init_model_path: 初始模型路径
            lr: 学习率
            optim_name: 优化器类名称
            cuda: 是否需要使用cuda
        """

        self.torch = torch
        self.model_class = model_class #传入的model_class是callable对象
        self.init_model_path = init_model_path #
        self.lr = lr
        self.optim_name = optim_name
        self.cuda = cuda

        self._init_params()

    def _init_params(self):
        self.model = self.model_class()  # call方法，callable对象
        if self.init_model_path:  # 需要初始化模型路径
            self.model.load_state_dict(self.torch.load(
                self.init_model_path))  # load_state_dict用于将预训练的参数权重加载到新的模型之中，load使用pickle的unpickling功能将pickle对象文件反序列化到内存。此功能还可以有助于设备加载数据。

        if self.cuda and self.torch.cuda.is_available():  # is_available看你电脑的 GPU 能否被 PyTorch 调用
            self.model = self.model.cuda()  #

        self.optimizer = getattr(self.torch.optim,
                                 self.optim_name)(self.model.parameters(),
                                                  lr=self.lr)  # self.torch.optim对象的self.optim_name属性

    def update_grads(self, grads):  #
        self.optimizer.zero_grad() #把梯度置零，也就是把loss关于weight的导数变成0

        for k, v in self.model.named_parameters():  #
            v.grad = grads[k].type(v.dtype) #

        self.optimizer.step() #

    def update_params(self, params): #

        for k, v in self.model.named_parameters(): #
            v[:] = params[k]

        return self.model

    def load_model(self, path, force_reload=False): #
        if force_reload is False and self.load_from_path == path:
            return

        self.load_from_path = path
        self.model.load_static_dict(self.torch.load(path))

    def save_model(self, path):
        base = os.path.dirname(path)
        if not os.path.exists(base):
            os.makedirs(base)

        self.torch.save(self.model.state_dict(), path)

        return path


class BaseBackend(ABC): #抽象类
    @abstractmethod
    def mean(self, data):
        data = np.array(data)

        return data.mean(axis=0) #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵


class NumpyBackend(BaseBackend):
    def mean(self, data):
        return super().mean(data=data)


class PytorchBackend(BaseBackend): #自定义pytorch通信后端，分布式通信过程主要是完成模型训练过程中参数信息的传递
    def __init__(self, torch, cuda=False):
        self.torch = torch
        if cuda:
            if self.torch.cuda.is_available():
                self.cuda = True #使用gpu
        else:
            self.cuda = False #使用cpu

    def mean(self, data, dim=0):
        return self.torch.tensor(
            data,
            device=self.torch.cuda.current_device() if self.cuda else None,
        ).mean(dim=dim)

    def sum(self, data, dim=0):
        return self.torch.tensor(
            data,
            device=self.torch.cuda.current_device() if self.cuda else None,
        ).sum(dim=dim)

    def _check_model(self, model): #检查模型
        if not isinstance(model, PytorchModel):
            raise ValueError(
                "model must be type of PytorchModel not {}".format(
                    type(model)))

    def update_grads(self, model, grads): #更新梯度
        self._check_model(model=model)
        return model.update_grads(grads=grads)

    def update_params(self, model, params): #更新参数
        self._check_model(model=model)
        return model.update_params(params=params)

    def load_model(self, model, path, force_reload=False): #加载模型
        self._check_model(model=model)
        return model.load_model(path=path, force_reload=force_reload)

    def save_model(self, model, path): #存储模型
        self._check_model(model=model)
        return model.save_model(path)


class Aggregator(object): #聚合器类
    def __init__(self, model, backend):  # backend ：指定当前进程要使用的通信后端
        self.model = model
        self.backend = backend  #通信后端


class FederatedSGD(Aggregator):
    def __init__(self, model, framework=None): #构造函数
        self.framework = framework or getattr(model, 'framework') #初始化模型的框架

        if framework is None or framework == 'numpy':#使用numpy框架或pytorch框架
            backend = NumpyBackend
        elif framework == 'pytorch':
            backend = PytorchBackend(torch=torch)
        else:
            raise ValueError(
                'Framework {} is not supported!'.format(framework))

        super().__init__(model, backend)

    def aggregate_grads(self, grads): #将模型梯度聚合到模型中
        """Aggregate model gradients to models.

        Args:
            data: a list of grads' information
                item format:
                    {
                        'n_samples': xxx,
                        'named_grads': xxx,
                    }
        """
        self.backend.update_grads(self.model,
                                  grads=aggregate_grads(grads=grads,
                                                        backend=self.backend)) #

    def save_model(self, path):
        return self.backend.save_model(self.model, path=path)  #

    def load_model(self, path, force_reload=False):
        return self.backend.load_model(self.model,
                                       path=path,
                                       force_reload=force_reload) #加载模型

    def __call__(self, grads): #传入梯度参数
        """Aggregate grads.

        Args:
            grads -> list: grads is a list of either the actual grad info
            or the absolute file path  of grad info. 实际的梯度信息或者梯度信息的绝对路径的列表
        """
        if not grads:
            return

        if not isinstance(grads, list): #如果梯度不是列表，则报错
            raise ValueError('grads should be a list, not {}'.format(
                type(grads)))

        actual_grads = grads

        return self.aggregate_grads(grads=actual_grads)
