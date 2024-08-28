import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import mne
import torch.optim as optim
from epochsMaker import import_EEG, EEG_to_epochs
from shallowConEx import ShallowConvNet

# EEG 데이터 불러오기
file_name = '[HRS]MI_four_1.txt'
eeg_array, label_array = import_EEG(file_name)
epochs = EEG_to_epochs(eeg_array, label_array)

event_id = {'Rest': 0, 'Right Hand': 1, 'Left Hand': 2, 'Feet': 3}  # 예시, 실제 데이터셋에 맞게 설정
events = epochs.events
event_name_map = {code: name for name, code in event_id.items()}

# 필터링 추가
# filter 제거
#epochs_filtered = epochs.copy().filter(l_freq=0.5, h_freq=40.0)
data_3d = epochs.get_data()
print(data_3d.shape)
labels = epochs.events[:, -1]

# ShallowConvNet 클래스 정의 (주어진 코드 그대로 사용)
class ShallowConvNet(nn.Module):
    def __init__(
            self,
            num_channels,  # 입력 데이터의 채널 개수
            output_dim=4,  # 츨력 차원 수
            dropout_prob=0.3,   # 드롭아웃 확률
            last_size=394   # fully connected 마지막 레이어의 입력 크기 설정
    ):
        super(ShallowConvNet, self).__init__()

        self.last_size = last_size
        self.num_channels = num_channels

        self.conv_temp = nn.Conv2d(1, 40, kernel_size=(1, 25)) 
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(num_channels, 1), bias=False) 
        self.batchnorm1 = nn.BatchNorm2d(40, momentum=0.1, affine=True, eps=1e-5)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)) 
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(last_size, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(1)
        
        x = self.conv_temp(input)
        x = self.conv_spat(x)
        x = self.batchnorm1(x)
        x = torch.square(x)
        x = self.avgpool1(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout1(x)
        output = self.fc(x)
        
        return output

# 데이터셋과 데이터로더 준비
def prepare_dataloader(epochs, labels, batch_size=32):
    dataset = TensorDataset(torch.tensor(data_3d, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 모델 초기화 및 학습 설정
def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # View as (batch_size, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    print('Finished Training')

# 모델 저장 함수
def save_model(model, optimizer, PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

# 모델 불러오기 함수
def load_model(model, optimizer, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# EEG 데이터 준비
# 하이퍼파라미터 설정
num_channels = 31
output_dim = 4
dropout_prob = 0.3
last_size = 394
batch_size = 8
num_epochs = 40
learning_rate = 0.001
repeat_times = 5  # 반복 횟수 설정

# 데이터로더 준비
dataloader = prepare_dataloader(data_3d, labels, batch_size=batch_size)


# 모델 초기화
model = ShallowConvNet(num_channels, output_dim, dropout_prob, last_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
# 반복 학습
PATH = './shallowconvnet.pth'

for i in range(repeat_times):
    print(f"Training iteration {i+1}/{repeat_times}")

    # 모델 학습
    train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate)

    # 모델 저장
    save_model(model, optimizer, PATH)

    # 모델 불러오기
    load_model(model, optimizer, PATH)

    # 모델 학습을 계속 진행할 수 있도록 옵티마이저의 학습률 감소 등 조정 가능
    # for example, adjust learning rate or other parameters if needed

print("All training iterations completed.")
