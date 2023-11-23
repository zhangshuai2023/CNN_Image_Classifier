# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

# 数据集读取并预处理
data_batch = []
for i in range(5):
    data_batch.append('data_batch_' + str(i+1))

test_batch = []
test_batch.append('test_batch')

key_file = {
    'train_img': data_batch,
    'train_label': data_batch,
    'test_img': test_batch,
    'test_label': test_batch
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/dataset_cifar_10.pkl"

train_num = 50000  # 训练数据量 
test_num = 10000 # 测试数据量  
img_dim = (3, 32, 32) # 32x32 彩色图像
img_size = 3072 # 一张图片共有 3072 个 0~255 值

        
def _load_label(file_names):
    a = []
    labels = np.array(a).astype(np.int32)
    for file_name in file_names:
        file_path = dataset_dir + "/" + file_name
        
        print("Extracting labels from " + file_name)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        labels = np.append(labels, dict[b'labels'][:]) # 从训练数据中提取标签数据

    print("Done")
    
    return labels

def _load_img(file_names):
    a = []
    data = np.array(a).astype(np.int32)
    for file_name in file_names:
        file_path = dataset_dir + "/" + file_name

        print("Extracting images from " + file_name)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        data = np.append(data, dict[b'data'][:]) # 从训练数据中提取图片数据
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}

    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_dataset():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_dataset(normalize=True, flatten=True, one_hot_label=False):
    """读入CIFAR_10数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下, 标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_dataset()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 32, 32)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_dataset()


# (train_img, train_label), (test_img, test_label) = load_dataset(False, False, False)
# print(str(train_img[0][0][:]))
# print(str(train_img.shape))
# print(str(test_img.shape))
# for label in train_label:
#     if label < 1:
#         print(str(label))

# print(str(train_label[:50]))
# print(str(test_label[:50]))

'''
# 测试数据读取情况
(train_img, train_label), (test_img, test_label) = load_dataset(False, False, False)

index = 0 #显示第一张图片
pic = train_img[index][0]
print(str(pic.shape))
print(pic)
print(train_label[index]) # 打印标签
import matplotlib.pyplot as plt
plt.imshow(pic)
plt.show()

'''