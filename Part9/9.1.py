# %matplotlib inline
import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img = Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)

# 定义绘图函数，本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
    plt.show()

# 左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 改变图像亮度
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

# 改变图像色调
apply(img, torchvision.transforms.ColorJitter(hue=0.5))

# 改变对比度
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 同时改变
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 叠加起来
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

all_imges = torchvision.datasets.CIFAR10(train=True, root="../Datasets", download=True)
# all_imges的每⼀个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)

# 将⼩批量图像转成PyTorch需要的格式
flip_aug = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
no_aug = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root="../Datasets"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# 训练函数，本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

train_with_data_aug(flip_aug, no_aug)