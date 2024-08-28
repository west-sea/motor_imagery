import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

# getOtherData에서 get_data_2a 함수 import
from getOtherData import get_data_2a

fs = 250                 # sampling frequency
channel = 22             # number of electrode
num_input = 1            # number of channel picture (for EEG signal is always : 1)
num_class = 4            # number of classes (left hand, right hand, foot, tongue)
signal_length = 1875     # number of sample in each trial (7.5 seconds * 250 Hz)

F1 = 8                   # number of temporal filters
D = 3                    # depth multiplier (number of spatial filters)
F2 = D * F1              # number of pointwise filters

device = 'cpu'
kernel_size_1 = (1, round(fs / 2))
kernel_size_2 = (channel, 1)
kernel_size_3 = (1, round(fs / 8))
kernel_size_4 = (1, 1)

kernel_avgpool_1 = (1, 4)
kernel_avgpool_2 = (1, 8)
dropout_rate = 0.2

ks0 = int(round((kernel_size_1[0] - 1) / 2))
ks1 = int(round((kernel_size_1[1] - 1) / 2))
kernel_padding_1 = (ks0, ks1 - 1)
ks0 = int(round((kernel_size_3[0] - 1) / 2))
ks1 = int(round((kernel_size_3[1] - 1) / 2))
kernel_padding_3 = (ks0, ks1)

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1
        self.conv2d = nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D * F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout_rate)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d(D * F1, D * F1, kernel_size_3,
                                                padding=kernel_padding_3, groups=D * F1)
        self.Separable_conv2D_point = nn.Conv2d(D * F1, F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(F2)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)
        # layer 4
        self.Flatten = nn.Flatten()
        #self.Dense = nn.Linear(F2 * round(signal_length / 32), num_class)
        #self.Softmax = nn.Softmax(dim=1)
        
        # Calculate the input size for the Dense layer
        final_output_size = F2 * ((signal_length // (kernel_avgpool_1[1] * kernel_avgpool_2[1])))
        self.Dense = nn.Linear(final_output_size, num_class)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x))  # .relu()
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
    acc_train = Accuracy(task="multiclass", num_classes=num_class).to(device)

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
