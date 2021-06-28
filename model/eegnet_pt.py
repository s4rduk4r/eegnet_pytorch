'''
    EEGNet PyTorch implementation
    Original implementation - https://github.com/vlawhern/arl-eegmodels
    Original paper: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    ---
    EEGNet Parameters:

      nb_classes      : int, number of classes to classify
      Chans           : number of channels in the EEG data
      Samples         : sample frequency (Hz) in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. 
                        ARL recommends to set this parameter to be half of the sampling rate. 
                        For the SMR dataset in particular since the data was high-passed at 4Hz ARL used a kernel length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from .SeparableConv import SeparableConv1d

import torch.optim as optim

class EEGNet(nn.Module):
    def __init__(self, nb_classes: int, Chans: int = 64, Samples: int = 128,
                 dropoutRate: float = 0.5, kernLength: int = 63,
                 F1:int = 8, D:int = 2):
        super().__init__()

        F2 = F1 * D

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) # (B, F2, Samples - Chans + 1, 1)
        # In: (B, F2, Samples - Chans + 1, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.avg_pool = nn.AvgPool1d(4)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.conv3 = SeparableConv1d(F2, F2, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(F2)
        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
        self.avg_pool2 = nn.AvgPool1d(8)
        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        self.fc = nn.Linear(F2 * ((Samples - Chans + 1) // 32), nb_classes)

    def forward(self, x: torch.Tensor):
        # Block 1
        y1 = self.conv1(x)
        #print("conv1: ", y1.shape)
        y1 = self.bn1(y1)
        #print("bn1: ", y1.shape)
        y1 = self.conv2(y1)
        #print("conv2", y1.shape)
        y1 = F.relu(self.bn2(y1))
        #print("bn2", y1.shape)
        y1 = self.avg_pool(y1)
        #print("avg_pool", y1.shape)
        y1 = self.dropout(y1)
        #print("dropout", y1.shape)

        # Block 2
        y2 = self.conv3(y1)
        #print("conv3", y2.shape)
        y2 = F.relu(self.bn3(y2))
        #print("bn3", y2.shape)
        y2 = self.avg_pool2(y2)
        #print("avg_pool2", y2.shape)
        y2 = self.dropout(y2)
        #print("dropout", y2.shape)
        y2 = torch.flatten(y2, 1)
        #print("flatten", y2.shape)
        y2 = self.fc(y2)
        #print("fc", y2.shape)

        return y2

'''
# Fitness hyperparams
LEARNING_RATE = 1e-2
EPOCHS_MAX = 5

# EEGNet hyperparams
NB_CLASSES = 4
KERNEL_LENGTH = 63
CHANNELS = 64
SAMPLES = 128
F1 = 8
D = 2

# Start of a program
model = EEGNet(NB_CLASSES, kernLength=KERNEL_LENGTH)
'''
'''# DEBUG
x = torch.rand((1, 64, 128, 1))
y = model(x)
print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")
'''

# Data prepare
'''
Dataset paper:
    https://www.nature.com/articles/sdata2018211
Datasets:
    https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698

# Stub
train, test, validation = [0, 1, 2]

# Loss function
loss = nn.CrossEntropyLoss()

# Optimizer
optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Fit cycle
#for epoch in range(EPOCHS_MAX):
    # Get data
    #x = torch.randn(1, )
'''
