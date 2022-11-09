import torch
from torch import nn

# 模型的读写
x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)

# 模型的读取
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
net.state_dict()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

'''
# 保存
torch.save(model.state_dict(), PATH) # 推荐的⽂件后缀名是pt或pth
# 加载
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))

# 保存
torch.save(model, PATH)
# 加载
model = torch.load(PATH)
'''

X = torch.randn(2, 3)
Y = net(X)
PATH = "./net.pt"
torch.save(net.state_dict(), PATH)
net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)