import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from getOtherData import get_data_2a

F1 = 8 # number of temporal filters
D = 3 # depth
C = 22 # channel
p = 0.25 # dropout rate / within-subject classification: 0.25, cross-subject: 0.5 / check 2.2.1 in paper
F2 = D * F1 # number of pointwise filters
T = 1750 # time points / 7 * 250Hz
N = 4 # number of classes

##################
# conv2d layer에 padding을 주면 T 숫자가 커져야함, 일단 padding 제거
##################
#T = 2060     # number of sample in each trial (7.5 seconds * 250 Hz)
device = 'cpu'

##################
# group: 
    # = 1 -> all input이 all output에 합성
    # = in_channel 개수 -> input이 out_C / in_C 에 합성
##################

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1
        self.conv2d = nn.Conv2d(1, F1, (1, 64), bias=False) # bias: Mode = same
        self.Batch_normalization_1 = nn.BatchNorm2d(F1) # 
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D * F1, (C, 1), groups=F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D * F1)
        self.Elu = nn.ELU() # Activation ELU, used 2 times
        self.Average_pooling2D_1 = nn.AvgPool2d((1, 4))
        self.Dropout = nn.Dropout2d(p)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1)
        self.Separable_conv2D_point = nn.Conv2d(D * F1, F2, (1, 1)) # (1, 1) from separable conv2d example
        self.Batch_normalization_3 = nn.BatchNorm2d(F2) # bias: 레이서 통과하고 뭔가를 추가?? 찾아보기/ 일반성 키우는 용
        self.Average_pooling2D_2 = nn.AvgPool2d((1, 8))
        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(F2 * ((T // 32)), N)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # unsqueeze: 아무 자리의 dimension에 1 추가 -> 하면은 8, 1, 22, 1875
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x))  # .relu() <-?
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)

        return y

model = EEGNet()

# 데이터 불러오기
# 받은 코드에서 data를 그대로 불러온 후, tensor 형태로 변환
# 차원 추가 -> (N, 1, 22, 1750)
# TensorDataset - 데이터셋 객체 생성 / DataLoader - 데이터 로드 배치 256

subject = 1
training = True
data, labels = get_data_2a(subject, training)
data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # (N, 1, 22, 1875)
labels = torch.tensor(labels, dtype=torch.long)  # (N,)

train_batch_size = 256

dataset = TensorDataset(data, labels)

data_loader = DataLoader(dataset,
                         batch_size=train_batch_size,
                         shuffle=True)

print("train batch size:", data_loader.batch_size,
      ", num of batch:", len(data_loader))

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    loss_train = AverageMeter()
    acc_train = Accuracy(task="multiclass", num_classes=N).to(device)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        loss_train.update(loss.item())
        acc_train(outputs, targets.int())

    return model, loss_train.avg, acc_train.compute().item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

num_epochs = 101
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.NAdam(model.parameters(), lr=0.01)

loss_train_hist = []
acc_train_hist = []

for epoch in range(num_epochs):
    model, loss_train, acc_train = train_one_epoch(model,
                                                   data_loader,
                                                   loss_fn,
                                                   optimizer)

    loss_train_hist.append(loss_train)
    acc_train_hist.append(acc_train)

    if (epoch % 10 == 5) or (epoch % 10 == 0):
        print(f'epoch {epoch}:')
        print(f' Loss= {loss_train:.4}, Accuracy= {int(acc_train * 100)}% \n')
