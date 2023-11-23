# coding: utf-8
import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定 失效，用下面代替
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:
    """简单的ConvNet

    conv - relu - pool - conv - relu - pool - affine - softmax
    
    Parameters
    ----------
    input_size : 输入大小 (cifar_10的情况下为3072) 3x32x32像素
    hidden_size_list : 隐藏层的神经元数量的列表 (e.g. [100, 100, 100])
    output_size : 输出大小 (cifar_10的情况下为10)0~9 的判定
    activation : 'relu' or 'sigmoid' 激活函数
    weight_init_std : 指定权重的标准差(e.g. 0.01)
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    use_dropout: 是否使用Dropout
    dropout_ration : Dropout的比例
    use_batchNorm: 是否使用Batch Normalization
    """
    def __init__(self, input_dim_1=(3, 32, 32), input_dim_2=(32, 16, 16), 
                 conv_param={'filter_num_1':32, 'filter_size_1':3, 'pad_1':1, 'stride_1':1, 
                             'filter_num_2':64, 'filter_size_2':3, 'pad_2':1, 'stride_2':1, },
                 hidden_size=128, output_size=10, weight_init_std=0.01, weight_decay_lambda = 0.1,
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False): #  用 He 初始值 sqr(2/n) 
        filter_num_1 = conv_param['filter_num_1']
        filter_size_1 = conv_param['filter_size_1']
        filter_pad_1 = conv_param['pad_1']
        filter_stride_1 = conv_param['stride_1']
        input_size_1 = input_dim_1[1] # 这里是32

        filter_num_2 = conv_param['filter_num_2']
        filter_size_2 = conv_param['filter_size_2']
        filter_pad_2 = conv_param['pad_2']
        filter_stride_2 = conv_param['stride_2']
        input_size_2 = input_dim_2[1] # 这里是16

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_ration = dropout_ration

        #第一次卷积和池化输出大小
        conv_output_size_1 = (input_size_1 - filter_size_1 + 2*filter_pad_1) / filter_stride_1 + 1 # 3通道卷积 3通道相加变成1通道 这里是 (32-3+2)/1+1 = 32 
        pool_output_size_1 = int(filter_num_1 * (conv_output_size_1/2) * (conv_output_size_1/2)) # 池化后边长为卷积的一半，32x16x16 =  8192

        #第二次卷积和池化输出大小
        conv_output_size_2 = (input_size_2 - filter_size_2 + 2*filter_pad_2) / filter_stride_2 + 1 # 32通道卷积 32通道相加变成1通道 这里是 (16-3+2)/1+1 = 16 
        pool_output_size_2 = int(filter_num_2 * (conv_output_size_2/2) * (conv_output_size_2/2)) # 池化后边长为卷积的一半，64x8x8 =  4096

        self.weight_decay_lambda = weight_decay_lambda
        
        # 初始化权重
        # self.params = {}
        # self.__init_weight(weight_init_std)

        #这可选的 0.01, Xavier, He
        self.params = {}
        #self.load_params()
        scale_list = self.__init_weight_scale(weight_init_std)
        
        self.params['W1'] = scale_list[0] * \
                            np.random.randn(filter_num_1, input_dim_1[0], filter_size_1, filter_size_1) # 初始化卷积核 4维 形状是(32, 3, 3, 3)
        self.params['b1'] = np.zeros(filter_num_1)

        self.params['W2'] = scale_list[1] * \
                            np.random.randn(filter_num_2, input_dim_2[0], filter_size_2, filter_size_2) # 初始化卷积核 4维 形状是(64, 32, 3, 3)
        self.params['b2'] = np.zeros(filter_num_2)

        self.params['W3'] = scale_list[2] * \
                            np.random.randn(pool_output_size_2, hidden_size) # 2维 形状是(4096, 100)
        self.params['b3'] = np.zeros(hidden_size)

        self.params['W4'] = scale_list[3] * \
                            np.random.randn(hidden_size, output_size) # 2维 形状是(100, 10)
        self.params['b4'] = np.zeros(output_size)
        hidden_size_list = [32768, 16384, 128]
        if self.use_batchnorm:  # 第一层的 batchnorm
            self.params['gamma1'] = np.ones(hidden_size_list[0])
            self.params['beta1'] = np.zeros(hidden_size_list[0])

        if self.use_batchnorm:  # 第二层的 batchnorm
            self.params['gamma2'] = np.ones(hidden_size_list[1])
            self.params['beta2'] = np.zeros(hidden_size_list[1])
    
        if self.use_batchnorm:  # 第三层的 batchnorm
            self.params['gamma3'] = np.ones(hidden_size_list[2])
            self.params['beta3'] = np.zeros(hidden_size_list[2])

        # 生成层
        #hidden_size_list = [32768, 16384, 128]

        self.layers = OrderedDict() # 一个有顺序的字典，使用时按顺序传播
        # 卷积层1  =============
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride_1'], conv_param['pad_1'])
        
        # batchnorm层1  =============
        if self.use_batchnorm:  # 第一层的 batchnorm
            # self.params['gamma1'] = np.ones(hidden_size_list[0])
            # self.params['beta1'] = np.zeros(hidden_size_list[0])
            self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], 
                                                           self.params['beta1'])    
        # Relu层1  =============
        self.layers['Relu1'] = Relu()
        
        # dropout层1  =============
        if self.use_dropout:  # 第一层的 dropout
            self.layers['Dropout1'] = Dropout(dropout_ration)

        # 池化层1  =============
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2) # 池化2x2, 步长为2 
        
        # 卷积层2  =============
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride_2'], conv_param['pad_2'])
        
        # batchnorm层2  =============
        if self.use_batchnorm:  # 第二层的 batchnorm
            # self.params['gamma2'] = np.ones(hidden_size_list[1])
            # self.params['beta2'] = np.zeros(hidden_size_list[1])
            self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], 
                                                            self.params['beta2'])  
        # Relu层2  =============
        self.layers['Relu2'] = Relu()

        # dropout层2  =============
        if self.use_dropout:  # 第二层的 dropout
            self.layers['Dropout2'] = Dropout(dropout_ration)

        # 池化层2  =============
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2) # 池化2x2, 步长为2

        # 全连接层1  =============
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        
        # batchnorm层3  =============
        if self.use_batchnorm:  # 第三层的 batchnorm
            # self.params['gamma3'] = np.ones(hidden_size_list[2])
            # self.params['beta3'] = np.zeros(hidden_size_list[2])
            self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], 
                                                            self.params['beta3'])  
        
        # Relu层3  =============
        self.layers['Relu3'] = Relu()

        # dropout层3  =============
        if self.use_dropout:  # 第三层的 dropout
            self.layers['Dropout3'] = Dropout(dropout_ration)

        # 全连接层2  =============
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss() # softmax 前向传播、后向传播的函数, 计算出Loss
     
    # def __init_weight(self, weight_init_std):
    #     """设定权重的初始值

    #     Parameters
    #     ----------
    #     weight_init_std : 指定权重的标准差（e.g. 0.01）
    #         指定'relu'或'he'的情况下设定“He的初始值”
    #         指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    #     """
    #     all_size_list = [3072, 32768, 16384, 4096, 128, 10]
    #     for idx in range(1, len(all_size_list)):
    #         scale = weight_init_std
    #         if str(weight_init_std).lower() in ('relu', 'he'):
    #             scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
    #         elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
    #             scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
    #         self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
    #         self.params['b' + str(idx)] = np.zeros(all_size_list[idx]) 

    def __init_weight_scale(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [3072, 8192, 4096, 128, 10]
        scale_list = []
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            scale_list.append(scale)

        return scale_list

    def predict(self, x):   # 一层一层地前向传播
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t): # x 是输入的图像，函数内部的 y 是输出向量，t 是训练标签
        """求损失函数
        参数x是输入数据、t是训练标签
        """
        y = self.predict(x)

        #---这里加入正则化项
        weight_decay = 0  # 正则化 weight decay
        for idx in range(1, 5):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay # 用 y 经过 softmax 计算得到预测值，并计算与监督向量的损失

    def accuracy(self, x, t, batch_size=100):  # t 是标签
        if t.ndim != 1 : t = np.argmax(t, axis=1) # t的维度不为1，取每列的最大值，形成1维
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)): # 有几个批，每个批次计算
            tx = x[i*batch_size:(i+1)*batch_size] # x 截取 这个批次的
            tt = t[i*batch_size:(i+1)*batch_size] # t 截取
            y = self.predict(tx) # 预测这个批次
            y = np.argmax(y, axis=1) # 预测的结果
            acc += np.sum(y == tt) # 100 个中所有预测正确的相加 
        
        return acc / x.shape[0] #总的正确率

    def numerical_gradient(self, x, t):  # 数值梯度 用数值微分求梯度
        """求梯度（数值微分）

        Parameters
        ----------
        x : 输入数据
        t : 训练标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t): # 反向传播法
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 训练标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t) # 正向传播

        # backward
        dout = 1
        dout = self.last_layer.backward(dout) # 最后一层反向传播

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout) # 各层反向传播

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db     
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        if self.use_batchnorm:  # 第一层的 batchnorm
            grads['gamma1'], grads['beta1'] = self.layers['BatchNorm1'].dgamma, self.layers['BatchNorm1'].dbeta
            grads['gamma2'], grads['beta2'] = self.layers['BatchNorm2'].dgamma, self.layers['BatchNorm2'].dbeta
            grads['gamma3'], grads['beta3'] = self.layers['BatchNorm3'].dgamma, self.layers['BatchNorm3'].dbeta

        return grads
        
    def save_params(self, file_name="params.pkl"): # 记录参数
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        # for i, key in enumerate(['Conv1','Conv2', 'Affine1', 'Affine2']):
        #     self.layers[key].W = self.params['W' + str(i+1)]
        #     self.layers[key].b = self.params['b' + str(i+1)]
        
        # if self.use_batchnorm:  # 第一层的 batchnorm
        #     for i, key in enumerate(['BatchNorm1','BatchNorm2', 'BatchNorm3']):
        #         self.layers[key].gamma = self.params['gamma' + str(i+1)]
        #         self.layers[key].beta = self.params['beta' + str(i+1)]

    