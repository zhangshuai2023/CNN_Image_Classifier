# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.optimizer import *

class Trainer:
    """进行神经网络的训练的类
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1) # 每轮迭代次数 = 训练数据集大小/批次大小
        self.max_iter = int(epochs * self.iter_per_epoch)  # 最大迭代次数 = 轮数 x 每轮迭代次数
        self.current_iter = 0 # 当前迭代次数
        self.current_epoch = 0 # 当前轮数
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size) # 随机选择 100 个
        x_batch = self.x_train[batch_mask] # 训练数据提取
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch) # 得到梯度，100 个样本 前向传播、后向传播 1 次  mini-batch 学习 计算交叉熵损失
        self.optimizer.update(self.network.params, grads) # 用梯度，按Adam 方法进行参数更新 卷积网络的 参数列表 params
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss) # 计算损失的
        if self.verbose: print("train loss:" + str(loss)) # 输出损失
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1 # 当前轮数
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch #  每轮评估的样本数  这里每轮评估不是所有样本，而是抽1000个出来评估
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc) # 保存训练精度
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter): # 最大迭代次数
            self.train_step() # 分步训练

        test_acc = self.network.accuracy(self.x_test, self.t_test) # 最终测试精度 是所有测了一遍

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

    def save_results(self, file_name="results.pkl"): # 记录训练过程损失和精度
        results = {}
        results['train_loss_list'] = self.train_loss_list
        results['train_acc_list'] = self.train_acc_list
        results['test_acc_list'] = self.test_acc_list
        import pickle
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)

    def load_results(self, file_name="results.pkl"):
        import pickle
        with open(file_name, 'rb') as f:
            results = pickle.load(f)
        self.train_loss_list = results['train_loss_list']
        self.train_acc_list = results['train_acc_list']
        self.test_acc_list = results['test_acc_list']