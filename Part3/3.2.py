# %matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后⼀次可能不⾜⼀个batch
        yield features.index_select(0, j), labels.index_select(0, j)

# 定义模型
def linreg(X, w, b): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    return torch.mm(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    # 注意这⾥返回的是向量, 另外, pytorch⾥的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这⾥更改param时⽤的param.data



# 生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

# 在../d2lzh_pytorch⾥⾯添加上⾯两个函数后就可以这样导⼊
import sys
sys.path.append("..")
from d2lzh_pytorch import *
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show() # 展示函数

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.double)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 训练模型
batch_size = 10
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):  # 训练模型⼀共需要num_epochs个迭代周期
    # 在每⼀个迭代周期中，会使⽤训练数据集中所有样本⼀次（假设样本数能够被批量⼤⼩整除）。X
    # 和y分别是⼩批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关⼩批量X和y的损失
        l.backward()  # ⼩批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使⽤⼩批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
print(true_w, '\n', w)
print(true_b, '\n', b)