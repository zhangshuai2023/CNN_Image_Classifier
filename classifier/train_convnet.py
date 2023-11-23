# coding: utf-8
import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定  失效，用下面代替
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import numpy as np
import matplotlib.pyplot as plt
from dataset.CIFAR_10 import load_dataset
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from common.util import shuffle_dataset

# 读入数据
(x_train, t_train), (x_test, t_test) = load_dataset(flatten=False)

# 处理花费时间较长的情况下减少数据 
# x_train, t_train = shuffle_dataset(x_train, t_train) # 打乱数据集，不需要，后边用随机批量训练的
# x_test, t_test = shuffle_dataset(x_test, t_test)
x_train, t_train = x_train[:10000], t_train[:10000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 60 #原 20  # 训练轮数

# 加入优化过拟合的方法=================================
# 设定 weight decay 的强度
weight_decay_lambda = 0.001
# 设定是否使用Dropuout，以及比例 
use_dropout = True  # 不使用Dropout的情况下为False
dropout_ration = 0.3
# 设定是否使用 batchnorm 
use_batchnorm = True
# ====================================================
network = SimpleConvNet(input_dim_1=(3,32,32), input_dim_2=(32, 16, 16), 
                        conv_param = {'filter_num_1':32, 'filter_size_1':3, 'pad_1':1, 'stride_1':1, 
                                    'filter_num_2':64, 'filter_size_2':3, 'pad_2':1, 'stride_2':1, },
                        hidden_size=128, output_size=10, weight_init_std= 'he', weight_decay_lambda = weight_decay_lambda,
                        use_dropout = use_dropout, dropout_ration = dropout_ration, use_batchnorm=use_batchnorm) # 加入
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
trainer.save_results()

# 保存参数
network.save_params("params.pkl")
print("Saved Network Parameters!")

#print(trainer.max_iter)
# 绘制图形
plt.figure(1)
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
#plt.show()

plt.figure(2)
markers2 = {'train': 'o'}
x_loss = np.arange(trainer.max_iter)
plt.plot(x_loss, trainer.train_loss_list, marker='', label='train', markevery=2)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3.0)
plt.legend(loc='upper right')
plt.show()