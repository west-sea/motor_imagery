import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from getOtherData2 import get_data_2a

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import seaborn as sns

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

subject = 10 # 실험자 번호
dataT, labelsT, dataE, labelsE = get_data_2a(subject)

dataT = torch.tensor(dataT, dtype=torch.float32).unsqueeze(1)
labelsT = torch.tensor(labelsT, dtype=torch.long)
dataE = torch.tensor(dataE, dtype=torch.float32).unsqueeze(1)
labelsE = torch.tensor(labelsE, dtype=torch.long)

train_batch_size = 256

train_dataset = TensorDataset(dataT, labelsT)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = TensorDataset(dataE, labelsE)
test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

print("Train batch size:", train_loader.batch_size, ", number of batches:", len(train_loader))
print("Test batch size:", test_loader.batch_size, ", number of batches:", len(test_loader))

#######################
# epoch: 전체 데이터의 학습 주기, 1 epoch면 1번 학습
# batch: 한 번의 테스트마다 주는 데이터의 size
# iteration: 1 epoch를 끝내기 위해 필요한 loop 횟수

# backpropagation 역전파: 출력값에 대한 입력값의 기울기를 출력층 layer에서부터 계산하여 거꾸고 전파시키는 것
# -> 출력층의 ouput에 대한 입력층에서의 input의 기울기(미분값)을 구할 수 있음
# => parameter와 layer가 많을 때, 가중치 w와 b를 학습시키기 어렵다는 문제 해결!
# => layer에서 기울기 값 구함 -> gradient descent를 이용하여 가중치 update

# gradient cliping: gradient 폭주 해결책
# -> backpropagation 떄 threshold를 넘지 못하도록 gradient 값을 자르는 방법
# gradient 소실 / 폭주: backpropagation에서 입력층으로 갈수록 기울기가 점점 작아짐 / 커짐 문제
#######################

# 한 epoch에 대한 학습 루프
def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train() # 모델을 훈련 모드로 설정
    loss_train = AverageMeter()
    acc_train = Accuracy(task="multiclass", num_classes=N).to(device)

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device) # 각 배치에 대해 입력과 타겟 데이터를 장치에 맞게 이동 
        outputs = model(inputs) # 모델을 통해 예측값 계산
        loss = loss_fn(outputs, targets) # 손실 계산


        loss.backward() # 역전파를 통해 손실에 대한 기울기 계산
        nn.utils.clip_grad_norm_(model.parameters(), 1) # 기울기 클리핑으로 기울기 폭발 방지
        optimizer.step() # optimizer를 통해 모델 가중치 업데이트 & 초기화
        optimizer.zero_grad()

        loss_train.update(loss.item())
        acc_train(outputs, targets)

    return model, loss_train.avg, acc_train.compute().item()

def evaluate(model, test_loader, loss_fn):
    model.eval()  # 모델을 평가 모드로 전환
    loss_test = AverageMeter()  # 손실을 추적하기 위한 AverageMeter 인스턴스 생성
    acc_test = Accuracy(task="multiclass", num_classes=N).to(device)  # 정확도를 계산하기 위한 Accuracy 인스턴스 생성

    with torch.no_grad():  # 평가 모드에서는 기울기를 계산하지 않도록 no_grad 블록 사용
        for inputs, targets in test_loader:  # 테스트 데이터 로더에서 입력과 타겟 데이터를 가져옴
            inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 장치(CPU 또는 GPU)로 이동
            outputs = model(inputs)  # 모델을 사용하여 예측값 계산
            loss = loss_fn(outputs, targets)  # 예측값과 타겟을 사용하여 손실 계산

            loss_test.update(loss.item())  # 손실 값을 업데이트
            acc_test(outputs, targets)  # 정확도 업데이트

    return loss_test.avg, acc_test.compute().item()  # 평균 손실과 정확도 반환

# confusion matrix 평가 추가
def plot_confusion_matrix(model, data_loader):
    model.eval()  # 모델을 평가 모드로 전환
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Left Hand', 'Right Hand', 'Both Feet', 'Tongue'], yticklabels=['Left Hand', 'Right Hand', 'Both Feet', 'Tongue'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class AverageMeter(object):
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

num_epochs = 31
loss_fn = nn.CrossEntropyLoss().to(device)

#################
# optimizer: gradient를 구해서, weight(가중치) 값을 변화시켜주는 역할
# optimizer 종류
# SGD: 가장 기본 / gradient가 작아질수록 loss의 최소값에 도달할 것이다! 이용
# momemtum: 언덕이 여러 개인 graph일 점 감안 -> 관성을 주어 local min을 넘어갈 수 있도록
# RMSprop: SGD에서 underfitting 일어날 때마다 lr 값 줄이기 & 일정한 비율로 step 조절
# Adam: SGD + momentum + RMSprop
#################
optimizer = optim.NAdam(model.parameters(), lr=0.01)
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)


# Train the model using T data
loss_train_hist = []
acc_train_hist = []

# Evaluate the model using E data first
loss_test, acc_test = evaluate(model, test_loader, loss_fn)
print(f'E Test Loss before: {loss_test:.4f}, E Test Accuracy before: {acc_test:.4f}')


### original training without k-fold

'''results = {}

for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # 데이터셋 분할
    train_subsampler = Subset(train_dataset, train_ids)
    test_subsampler = Subset(train_dataset, test_ids)

    train_loader = DataLoader(train_subsampler, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_subsampler, batch_size=train_batch_size, shuffle=False)

    # 모델 초기화
    model = EEGNet().to(device)
    optimizer = optim.NAdam(model.parameters(), lr=0.01)

    # 모델 훈련
    for epoch in range(num_epochs):
        model, loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {loss_train:.4f}, Train Accuracy: {acc_train:.4f}')

    # 모델 평가
    loss_test, acc_test = evaluate(model, test_loader, loss_fn)
    print(f'Test Loss: {loss_test:.4f}, Test Accuracy: {acc_test:.4f}')

    results[fold] = {'loss': loss_test, 'accuracy': acc_test}

# 각 폴드의 결과 출력
print('--------------------------------')
print('K-FOLD CROSS VALIDATION RESULTS FOR 5 FOLDS')
print('--------------------------------')
sum_loss = 0.0
sum_acc = 0.0
for key, value in results.items():
    print(f'Fold {key}: Loss {value["loss"]}, Accuracy {value["accuracy"]}')
    sum_loss += value['loss']
    sum_acc += value['accuracy']
print(f'Average Loss: {sum_loss/k_folds}')
print(f'Average Accuracy: {sum_acc/k_folds}')'''

for epoch in range(num_epochs):
    model, loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer)
    loss_train_hist.append(loss_train)
    acc_train_hist.append(acc_train)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}:')
        print(f' Train Loss: {loss_train:.4f}, Train Accuracy: {acc_train:.4f}')

# Evaluate the model using E data
loss_test, acc_test = evaluate(model, test_loader, loss_fn)
print(f'E Test Loss after: {loss_test:.4f}, E Test Accuracy after: {acc_test:.4f}')
loss_test, acc_test = evaluate(model, train_loader, loss_fn)
print(f'T Train Loss after: {loss_test:.4f}, T Train Accuracy after: {acc_test:.4f}')

# Confusion matrix
plot_confusion_matrix(model, test_loader)

'''plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_train_hist, label='Train Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(acc_train_hist, label='Train Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')'''

plt.show()
