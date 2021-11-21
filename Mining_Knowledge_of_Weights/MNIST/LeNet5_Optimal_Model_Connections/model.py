import torch
from torch import nn
from layer import Convolution0, Convolution1, Fully_Connect0, Fully_Connect1


class LeNet5_800_500(nn.Module):
    def __init__(self, c_in=1, c_dims1=20, c_dims2=50, fc_in=800, fc_dims1=500, num_classes=10):
        super(LeNet5_800_500, self).__init__()
        self.c_in = c_in
        self.c_dims1 = c_dims1
        self.c_dims2 = c_dims2
        self.fc_in = fc_in
        self.fc_dims1 = fc_dims1
        self.layer0 = Convolution0(self.c_in, self.c_dims1)
        self.layer1 = Convolution1(self.c_dims1, self.c_dims2)
        self.layer2 = Fully_Connect0(self.fc_in, self.fc_dims1)
        self.layer3 = Fully_Connect1(self.fc_dims1, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, epoch):
        # 1
        out, c_gate0 = self.layer0(x, epoch)
        out = self.maxpool(out)
        # 2
        out, c_gate1 = self.layer1(out, epoch)
        out = self.maxpool(out)
        o = out.view(out.size(0), -1)
        # 3
        out, f_gate0 = self.layer2(o, epoch)
        out = self.relu(out)
        # 4
        out, f_gate1 = self.layer3(out, epoch)
        return out, c_gate0, c_gate1, f_gate0, f_gate1