# 시도 1) 실제 테스트 때 사용했던 모델.
#
# 주어졌던 신호 데이터들을 합치면 음성 신호와 같은 형태가 나올 것이라는 점에서 착안하여
# 6개 신호를 모두 합친 후 단일 신호를 생성한 뒤 STFT 알고리즘을 적용하여 스펙트로그램
# 이미지로 변경.
#
# 이후 음악 장르 분류 모델로 쓰이는 CRNN을 사용하여 스펙트로그램 데이터를 처리.
#
# dataset0 test 기준 macro F1 0.2176, dataset1 train 기준 macro F1 0.2870을 기록
# 했으나, dataset1 test 기준 macro F1이 0.0이 나왔음(!)
#
# EDA 할 시간도 없이 바로 아이디어 구현 및 제출 파일 준비하느라 제대로 된 성능 파악을
# 하지 못한 것이 패착이었던 것으로 생각.


import torch
import torch.nn as nn

from einops import rearrange


PRGKR_DATSET0 = {
    "n_fft": 34,            # dataset0: 696 -> 88 -> 44 -> 22 -> 11
    "class_num": 16
}

PRGKR_DATSET1 = {
    "n_fft": 34,            # dataset1: 800 -> 101 -> 50 -> 25 -> 12
    "class_num": 11
}

MINDBIG_DATSET0 = {
    "n_fft": 35,            # dataset0: 368 -> 46 -> 23 -> 11 -> 5
    "class_num": 15
}

MINDBIG_DATSET1 = {
    "n_fft": 34,            # dataset1: 252 -> 32 -> 16 -> 8 -> 4
    "class_num": 10
}


class SingleSignalSTFT(nn.Module):
    def __init__(self, n_fft=34):
        super(SingleSignalSTFT, self).__init__()
        self.n_fft = n_fft

    def forward(self, x):
        x = torch.sum(x, dim=-1)    # 6가지 신호를 하나의 신호로 합침
        x = torch.stft(x, n_fft=self.n_fft) # 신호에 STFT를 적용하여 스펙트로그램으로 변환
        
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel_size):
        super(ConvBlock, self).__init__()

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class RNNBlock(nn.Module):
    def __init__(self):
        super(RNNBlock, self).__init__()

        self.num_layers = 2
        self.h_0 = nn.Parameter(torch.rand(2*self.num_layers, 1, 256, requires_grad=True), requires_grad=True)
        self.BiGRU = nn.GRU(input_size=64, hidden_size=256, num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        h_0 = torch.zeros((2*self.num_layers, x.shape[0], 256)).cuda() # is not continguous 에러 해결
        h_0 += torch.broadcast_to(self.h_0, (2*self.num_layers, x.shape[0], 256))
        output, h_n = self.BiGRU(x, h_0)
        return output


class CRNN(nn.Module):
    def __init__(self, n_fft, class_num):
        super(CRNN, self).__init__()
        
        self.STFT = SingleSignalSTFT(n_fft=n_fft)
        
        self.Conv1 = ConvBlock(in_channels=2, out_channels=16, pool_kernel_size=(2, 2))
        self.Conv2 = ConvBlock(in_channels=16, out_channels=32, pool_kernel_size=(3, 2))
        self.Conv3 = ConvBlock(in_channels=32, out_channels=64, pool_kernel_size=(3, 2))

        self.BiGRU = RNNBlock()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 512))

        self.fc = nn.Linear(in_features=512, out_features=class_num)

    def forward(self, x):
        # 신호를 합쳐서 STFT Spectrogram으로 변환
        x = self.STFT(x)

        # Convolution
        x = rearrange(x, "batch freq time channel -> batch channel freq time")
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)

        # Recurrent
        x = rearrange(x, "batch channel freq time -> batch (freq time) channel")
        x = self.BiGRU(x)

        x = self.global_avg_pool(x)
        output = self.fc(torch.flatten(x, start_dim=1))

        return output

    def init_weights(self, init_only_fc=False):
        for m in self.modules():
            if isinstance(m, nn.Linear): # init dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif not init_only_fc:
                if isinstance(m, nn.BatchNorm2d): # init BN
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d): # init conv
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)
