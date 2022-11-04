import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
import os
import torchvision
import torchvision.transforms as transforms

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

def use_svg_display():
    # ⽤⽮量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺⼨
    plt.rcParams['figure.figsize'] = figsize

# 本函数已保存在d2lzh包中⽅便以后使⽤
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中⽅便以后使⽤
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 获取并读取Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不⽤额外的进程来加速读取数据
    else:
        num_workers = 4
    path, _ = os.path.split(os.getcwd())
    path = path + '\\' + 'Datasets'
    if os.path.exists(path) == False:
        os.mkdir(path)
    # 获取数据集
    # 数据传到Datasets文件夹中
    mnist_train = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

# 评价模型准确率。本函数已保存在d2lzh_pytorch包中⽅便以后使⽤。该函数将被逐步改进：它的完整实现将在“图像增⼴”⼀节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 训练模型。本函数已保存在d2lzh包中⽅便以后使⽤
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”⼀节将⽤到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
        epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))