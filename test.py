import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import mne
import torch.optim as optim
from sklearn.metrics import accuracy_score
from epochsMaker import import_EEG, EEG_to_epochs
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

# EEG 데이터 가져오기 및 전처리
file_name = '[HRS]MI_four_1.txt'
eeg_array, label_array = import_EEG(file_name)
epochs = EEG_to_epochs(eeg_array, label_array)

event_id = {'Rest': 0, 'Right Hand': 1, 'Left Hand': 2, 'Feet': 3}  
events = epochs.events
event_name_map = {code: name for name, code in event_id.items()}

epochs_filtered = epochs.copy().filter(l_freq=8, h_freq=35.0)
filt_raw = epochs_filtered.copy().filter(l_freq=1.0, h_freq=None)

ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(epochs_filtered)
ica.exclude = [1]
reconst_raw = epochs_filtered.copy()
ica.apply(reconst_raw)

data_3d = reconst_raw.get_data()  
print(data_3d.shape)
labels = epochs.events[:, -1]

# ShallowConvNet 모델 정의
class ShallowConvNet(nn.Module):
    def __init__(self, num_channels, output_dim=4, dropout_prob=0.3):
        super(ShallowConvNet, self).__init__()

        self.conv_temp = nn.Conv2d(1, 40, kernel_size=(1, 25)) 
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(num_channels, 1), bias=False) 
        self.batchnorm1 = nn.BatchNorm2d(40, momentum=0.1, affine=True, eps=1e-5)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)) 
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self._calc_fc_input_dim(num_channels), output_dim)

    def _calc_fc_input_dim(self, num_channels):
        x = torch.zeros((1, 1, num_channels, 6000))  # 실제 데이터와 유사한 크기
        x = self.conv_temp(x)
        x = self.conv_spat(x)
        x = self.batchnorm1(x)
        x = torch.square(x)
        x = self.avgpool1(x)
        return x.view(1, -1).size(1)  # fc 레이어 입력 크기 계산

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
        x = x.view(x.size(0), -1)  # Flatten
        output = self.fc(x)
        
        return output

# 데이터로더 준비
def prepare_dataloader(data, labels, batch_size=32):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 모델 학습 함수
def train_model(model, dataloader, num_epochs=40, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    print('Finished Training')

# 실제 학습 시작
num_channels = 31
output_dim = 4
dropout_prob = 0.3
batch_size = 8
num_epochs = 100
learning_rate = 0.001

dataloader = prepare_dataloader(data_3d, labels, batch_size=batch_size)
model = ShallowConvNet(num_channels, output_dim, dropout_prob)

train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate)
