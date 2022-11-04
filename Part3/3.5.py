import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了导⼊上层⽬录的d2lzh_pytorch
import d2lzh_pytorch as d2l
import os

# 本函数已保存在d2lzh包中⽅便以后使⽤
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中⽅便以后使⽤
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

path, _ = os.path.split(os.getcwd())
path = path + '\\' + 'Datasets'
if os.path.exists(path) == False:
    os.mkdir(path)
# 获取数据集
# 数据传到上一级Datasets文件夹中
mnist_train = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transforms.ToTensor())
#print(type(mnist_train))
#print(len(mnist_train), len(mnist_test))
'''
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
'''

# 读取⼩批量
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不⽤额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))