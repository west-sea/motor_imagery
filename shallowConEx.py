import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            num_channels,  # 입력 데이터의 채널 개수
            output_dim=4,  # 츨력 차원 수
            dropout_prob=0.3,   # 드롭아웃 확률
            last_size=394   # fully connected 마지막 레이어의 입력 크기 설정
    ):
        super(ShallowConvNet, self).__init__()

        # if pretrain True, modifies intermediate layer output values based on finetune & pretrain channels,
        # if pretrain False, modifies intermediate layer output values based on self.scale
        self.last_size = last_size

        self.num_channels = num_channels

        # 합성곱 레이어1, 입력 채널:1 (흑백), 출력 채널:40
        self.conv_temp = nn.Conv2d(1, 40, kernel_size=(1, 25)) 
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(num_channels, 1), bias=False) 
        
        # 배치 정규화: 입력 데이터를 정규분포에 가깝게 만든다!
        # momentum: 현재 배치의 통계값에 이전 배치들의 통계값을 얼마나 반영할지 설정, 보통 0.1
        # affine: 배치 레이어가 scale 조정 & shift 가능한지 설정
        # eps: 분모가 0이 되는 것을 방지하기 위해 추가하는 매우 작은 값
        self.batchnorm1 = nn.BatchNorm2d(40, momentum=0.1, affine=True, eps=1e-5) #배치 정규화? 레이어, 출력 채널:40, affine: 학습 가능 여부-참
        
        # 평균 폴링 레이어: 입력 데이터에서 지정된 영역(kernel)의 평균 계산
        # stride: 이동 보폭 크기 / 수평1, 수직15 픽셀씩 이동하며 평균 계산
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)) #평균 풀링 레이어, 
        
        # 드롭아웃: 과적합을 방지하기 위한 정규화 기법
        # 학습 중에 random하게 뉴런의 출력을 0으로 만듬 -> 특정 뉴런에 의존하지 않도록
        self.dropout1 = nn.Dropout(p=dropout_prob) #드롭아웃 레이어

        # self.conv_class = nn.Conv2d(200,2,kernel_size=(1,9))
        #self.flatten = nn.Flatten() #데이터를 1차원으로 평탄화하는 레이어
        #fc: fully connected layer, 입력크기:last_size, 출력차원:output_dim
        self.fc = nn.Linear(last_size, output_dim)  # input length 500->1080, 750->1760, 1000->2440, 1125 -> 2760
        self.softmax = nn.LogSoftmax(dim=1) #출력에 LogSoftmax를 적용 -> 확률 값으로 변환

    #forward 메서드: 실제 데이터가 주어졌을 때 네트워크의 계산 실행
    def forward(self, input):
        #입력 데이터가 3차원이면, 4차원으로 올리기
        if len(input.shape)==3:
            input = input.unsqueeze(1) 
        # print("input: ", input.shape)
        
        #밑에는 위에서 만들어둔 layer들에 데이터 통과시키는 거
        x = self.conv_temp(input)
        # print("conv_temp: ",x.shape)
        x = self.conv_spat(x)
        # print("spat_temp: ",x.shape)
        x = self.batchnorm1(x)
        x = torch.square(x)
        # print("b4avgpool: ", x.shape)
        x = self.avgpool1(x)
        x = torch.log(torch.clamp(x,min=1e-6))
        # print("avgpool: ", x.shape)
        x = self.dropout1(x)

        #x = self.flatten(x)
        # print(x.shape)
        output = self.fc(x)
        
        return output