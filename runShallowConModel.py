import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import mne
from epochsMaker import import_EEG, EEG_to_epochs
from shallowConEx import ShallowConvNet

##############################
# wrong code #
##############################


file_name = '[HRS]MI_four_1.txt'
eeg_array, label_array = import_EEG(file_name)
epochs = EEG_to_epochs(eeg_array, label_array)
# print(epochs.get_data().shape)

event_id = {'Rest': 0, 'Right Hand': 1, 'Left Hand': 2, 'Feet': 3}  # 예시, 실제 데이터셋에 맞게 설정
events = epochs.events
event_name_map = {code: name for name, code in event_id.items()}

#dataset = TensorDataset(torch.tensor(epochs['data'], dtype=torch.float32), torch.tensor(epochs['label'], dtype=torch.long))
#dataset = TensorDataset(torch.tensor(epochs.get_data(), dtype=torch.float32), torch.tensor(events[:, -1], dtype=torch.long))
# 데이터셋 생성 시 레이블을 2D 텐서로 변환
#labels_2d = events[:, -1][:, np.newaxis]  # 각 레이블을 새로운 축에 추가하여 2D로 만듦
#dataset = TensorDataset(torch.tensor(epochs.get_data(), dtype=torch.float32), torch.tensor(labels_2d, dtype=torch.long))
data_3d = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
print(data_3d.shape)
labels = epochs.events[:, -1]

dataset = TensorDataset(torch.tensor(data_3d, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
batch_size = 31
training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = ShallowConvNet(num_channels=31)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
timestamp = '2024'  # 적절한 타임스탬프로 변경해야 함
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

# 학습 루프
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    model.train()

    running_loss = 0.0

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:
            avg_loss = running_loss / 100
            print('  Batch {} loss: {:.4f}'.format(i + 1, avg_loss))
            writer.add_scalar('Loss/train', avg_loss, epoch * len(training_loader) + i + 1)
            running_loss = 0.0

    # 검증 데이터에 대한 평가 (생략)

    # 모델 저장
    torch.save(model.state_dict(), 'model_{}_{}.pt'.format(timestamp, epoch))

print('Training finished.')

