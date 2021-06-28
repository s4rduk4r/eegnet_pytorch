'''
Depthwise Separable Convolution
Paper - https://arxiv.org/pdf/1704.04861.pdf
'''
import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y

class SeparableConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y


'''
C_IN = 40
C_OUT = 10
KERNEL_SIZE = (3, 3)

sep_conv = SeparableConv2d(C_IN, C_OUT, KERNEL_SIZE)
x = torch.randn(1, C_IN, 256, 256)
y = sep_conv(x)
conv = nn.Conv2d(C_IN, C_OUT, KERNEL_SIZE)
y1 = conv(x)

print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")
print(f"y1.shape = {y1.shape}")
assert y.shape == y1.shape
'''
