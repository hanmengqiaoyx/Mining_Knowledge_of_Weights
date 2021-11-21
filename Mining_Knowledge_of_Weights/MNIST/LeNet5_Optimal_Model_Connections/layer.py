import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F


class Convolution0(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Convolution0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, 1, 5, 5))
        self.gate_weights00 = Parameter(torch.Tensor(16, in_features, 5, 5))
        self.gate_bn00 = nn.BatchNorm2d(16)
        self.gate_weights01 = Parameter(torch.Tensor(16, 16, 1, 1))
        self.gate_bn01 = nn.BatchNorm2d(16)
        self.gate_weights02 = Parameter(torch.Tensor(in_features, 16, 1, 1))
        self.gate_bn02 = nn.BatchNorm2d(in_features)
        self.dw_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_weights3 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.up_sample0 = Parameter(torch.Tensor(64, 32, 3, 1))
        self.up_bn00 = nn.BatchNorm2d(32)
        self.up_weights0 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(32)
        self.up_sample1 = Parameter(torch.Tensor(32, 16, 2, 1))
        self.up_bn10 = nn.BatchNorm2d(16)
        self.up_weights1 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(16)
        self.up_sample2 = Parameter(torch.Tensor(16, 8, 2, 1))
        self.up_bn20 = nn.BatchNorm2d(8)
        self.up_weights2 = Parameter(torch.Tensor(8, 16, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(8)
        self.gate_weights1 = Parameter(torch.Tensor(1, 8, 3, 3))
        self.gate_bn1 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights00, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights01, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights02, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights1, mode='fan_out', nonlinearity='relu')

    def forward(self, input, epoch):
        if self.training:
            if epoch < 4:
                self.weights.requires_grad = True
                self.gate_weights00.requires_grad = False
                self.gate_weights01.requires_grad = False
                self.gate_weights02.requires_grad = False
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.gate_weights1.requires_grad = False
                gate_prt = torch.zeros(self.out_features, 1).cuda()
                output = nn.functional.conv2d(input, self.weights, stride=1, bias=None)
                output = self.relu(output)
            elif epoch >= 4 and epoch % 2 == 0:
                self.weights.requires_grad = False
                self.gate_weights00.requires_grad = True
                self.gate_weights01.requires_grad = True
                self.gate_weights02.requires_grad = True
                self.dw_weights0.requires_grad = True
                self.dw_weights1.requires_grad = True
                self.dw_weights2.requires_grad = True
                self.dw_weights3.requires_grad = True
                self.up_sample0.requires_grad = True
                self.up_sample1.requires_grad = True
                self.up_sample2.requires_grad = True
                self.up_weights0.requires_grad = True
                self.up_weights1.requires_grad = True
                self.up_weights2.requires_grad = True
                self.gate_weights1.requires_grad = True
                weight = self.weights.view(self.out_features, self.in_features, 5, 5)
                layer = F.relu(self.gate_bn00(nn.functional.conv2d(weight, self.gate_weights00, stride=1, bias=None)))
                layer = F.relu(self.gate_bn01(nn.functional.conv2d(layer, self.gate_weights01, stride=1, bias=None)))
                layer = F.relu(self.gate_bn02(nn.functional.conv2d(layer, self.gate_weights02, stride=1, bias=None)))
                data = layer.view(1, 1, self.out_features, self.in_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))  # [1, 64, 2, 1]
                layer40 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer30, self.up_sample0, stride=2, bias=None)))
                layer41 = torch.cat((layer20, layer40), dim=1)
                layer42 = F.relu(self.up_bn01(nn.functional.conv2d(layer41, self.up_weights0, stride=1, padding=1, bias=None)))
                layer50 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer42, self.up_sample1, stride=2, bias=None)))
                layer51 = torch.cat((layer10, layer50), dim=1)
                layer52 = F.relu(self.up_bn11(nn.functional.conv2d(layer51, self.up_weights1, stride=1, padding=1, bias=None)))
                layer60 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer52, self.up_sample2, stride=2, bias=None)))
                layer61 = torch.cat((layer00, layer60), dim=1)
                layer62 = F.relu(self.up_bn21(nn.functional.conv2d(layer61, self.up_weights2, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn1(nn.functional.conv2d(layer62, self.gate_weights1, stride=1, padding=1, bias=None))) # [1, 1, 20, 1]
                gate_prt = layer_out.view(self.out_features, 1)
                gate = layer_out.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = nn.functional.conv2d(input, new_weights, stride=1, bias=None)
                output = self.relu(output)
            elif epoch >= 4 and epoch % 2 != 0:
                self.weights.requires_grad = True
                self.gate_weights00.requires_grad = False
                self.gate_weights01.requires_grad = False
                self.gate_weights02.requires_grad = False
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.gate_weights1.requires_grad = False
                weight = self.weights.view(self.out_features, self.in_features, 5, 5)
                layer = F.relu(self.gate_bn00(nn.functional.conv2d(weight, self.gate_weights00, stride=1, bias=None)))
                layer = F.relu(self.gate_bn01(nn.functional.conv2d(layer, self.gate_weights01, stride=1, bias=None)))
                layer = F.relu(self.gate_bn02(nn.functional.conv2d(layer, self.gate_weights02, stride=1, bias=None)))
                data = layer.view(1, 1, self.out_features, self.in_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))  # [1, 64, 2, 3]
                layer40 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer30, self.up_sample0, stride=2, bias=None)))
                layer41 = torch.cat((layer20, layer40), dim=1)
                layer42 = F.relu(self.up_bn01(nn.functional.conv2d(layer41, self.up_weights0, stride=1, padding=1, bias=None)))
                layer50 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer42, self.up_sample1, stride=2, bias=None)))
                layer51 = torch.cat((layer10, layer50), dim=1)
                layer52 = F.relu(self.up_bn11(nn.functional.conv2d(layer51, self.up_weights1, stride=1, padding=1, bias=None)))
                layer60 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer52, self.up_sample2, stride=2, bias=None)))
                layer61 = torch.cat((layer00, layer60), dim=1)
                layer62 = F.relu(self.up_bn21(nn.functional.conv2d(layer61, self.up_weights2, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn1(nn.functional.conv2d(layer62, self.gate_weights1, stride=1, padding=1, bias=None))) # [1, 1, 20, 1]
                gate_prt = layer_out.view(self.out_features, 1)
                gate = layer_out.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = nn.functional.conv2d(input, new_weights, stride=1, bias=None)
                output = self.relu(output)
        elif not self.training:
            if epoch < 4:
                gate_prt = torch.zeros(self.out_features, 1).cuda()
                output = nn.functional.conv2d(input, self.weights, stride=1, bias=None)
                output = self.relu(output)
            else:
                weight = self.weights.view(self.out_features, self.in_features, 5, 5)
                layer = F.relu(self.gate_bn00(nn.functional.conv2d(weight, self.gate_weights00, stride=1, bias=None)))
                layer = F.relu(self.gate_bn01(nn.functional.conv2d(layer, self.gate_weights01, stride=1, bias=None)))
                layer = F.relu(self.gate_bn02(nn.functional.conv2d(layer, self.gate_weights02, stride=1, bias=None)))
                data = layer.view(1, 1, self.out_features, self.in_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))  # [1, 64, 2, 3]
                layer40 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer30, self.up_sample0, stride=2, bias=None)))
                layer41 = torch.cat((layer20, layer40), dim=1)
                layer42 = F.relu(self.up_bn01(nn.functional.conv2d(layer41, self.up_weights0, stride=1, padding=1, bias=None)))
                layer50 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer42, self.up_sample1, stride=2, bias=None)))
                layer51 = torch.cat((layer10, layer50), dim=1)
                layer52 = F.relu(self.up_bn11(nn.functional.conv2d(layer51, self.up_weights1, stride=1, padding=1, bias=None)))
                layer60 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer52, self.up_sample2, stride=2, bias=None)))
                layer61 = torch.cat((layer00, layer60), dim=1)
                layer62 = F.relu(self.up_bn21(nn.functional.conv2d(layer61, self.up_weights2, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn1(nn.functional.conv2d(layer62, self.gate_weights1, stride=1, padding=1, bias=None)))  # [1, 1, 20, 1]
                gate_prt = layer_out.view(self.out_features, 1)
                gate = layer_out.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = nn.functional.conv2d(input, new_weights, stride=1, bias=None)
                output = self.relu(output)
        return output, gate_prt


class Convolution1(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Convolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 5, 5))
        self.gate_weights00 = Parameter(torch.Tensor(2*in_features, in_features, 5, 5))
        self.gate_bn00 = nn.BatchNorm2d(2*in_features)
        self.gate_weights01 = Parameter(torch.Tensor(2*in_features, 2*in_features, 1, 1))
        self.gate_bn01 = nn.BatchNorm2d(2*in_features)
        self.gate_weights02 = Parameter(torch.Tensor(in_features, 2*in_features, 1, 1))
        self.gate_bn02 = nn.BatchNorm2d(in_features)
        self.dw_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_weights3 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.dw_weights4 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.up_sample0 = Parameter(torch.Tensor(128, 64, 2, 2))
        self.up_bn00 = nn.BatchNorm2d(64)
        self.up_weights0 = Parameter(torch.Tensor(64, 128, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(64)
        self.up_sample1 = Parameter(torch.Tensor(64, 32, 2, 3))
        self.up_bn10 = nn.BatchNorm2d(32)
        self.up_weights1 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(32)
        self.up_sample2 = Parameter(torch.Tensor(32, 16, 3, 2))
        self.up_bn20 = nn.BatchNorm2d(16)
        self.up_weights2 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(16)
        self.up_sample3 = Parameter(torch.Tensor(16, 8, 2, 2))
        self.up_bn30 = nn.BatchNorm2d(8)
        self.up_weights3 = Parameter(torch.Tensor(8, 16, 3, 3))
        self.up_bn31 = nn.BatchNorm2d(8)
        self.gate_weights1 = Parameter(torch.Tensor(1, 8, 3, 3))
        self.gate_bn1 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights00, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights01, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights02, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights4, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights1, mode='fan_out', nonlinearity='relu')

    def forward(self, input, epoch):
        if self.training:
            if epoch < 4:
                self.weights.requires_grad = True
                self.gate_weights00.requires_grad = False
                self.gate_weights01.requires_grad = False
                self.gate_weights02.requires_grad = False
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.gate_weights1.requires_grad = False
                gate_prt = torch.zeros(self.out_features * self.in_features, 1).cuda()
                output = nn.functional.conv2d(input, self.weights, stride=1, bias=None)
                output = self.relu(output)
            elif epoch >= 4 and epoch % 2 == 0:
                self.weights.requires_grad = False
                self.gate_weights00.requires_grad = True
                self.gate_weights01.requires_grad = True
                self.gate_weights02.requires_grad = True
                self.dw_weights0.requires_grad = True
                self.dw_weights1.requires_grad = True
                self.dw_weights2.requires_grad = True
                self.dw_weights3.requires_grad = True
                self.dw_weights4.requires_grad = True
                self.up_sample0.requires_grad = True
                self.up_sample1.requires_grad = True
                self.up_sample2.requires_grad = True
                self.up_sample3.requires_grad = True
                self.up_weights0.requires_grad = True
                self.up_weights1.requires_grad = True
                self.up_weights2.requires_grad = True
                self.up_weights3.requires_grad = True
                self.gate_weights1.requires_grad = True
                weight = self.weights.view(self.out_features, self.in_features, 5, 5)
                layer = F.relu(self.gate_bn00(nn.functional.conv2d(weight, self.gate_weights00, stride=1, bias=None)))
                layer = F.relu(self.gate_bn01(nn.functional.conv2d(layer, self.gate_weights01, stride=1, bias=None)))
                layer = F.relu(self.gate_bn02(nn.functional.conv2d(layer, self.gate_weights02, stride=1, bias=None)))
                data = layer.view(1, 1, self.out_features, self.in_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 3, 1]
                layer50 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer51 = torch.cat((layer30, layer50), dim=1)
                layer52 = F.relu(self.up_bn01(nn.functional.conv2d(layer51, self.up_weights0, stride=1, padding=1, bias=None)))
                layer60 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer52, self.up_sample1, stride=2, bias=None)))
                layer61 = torch.cat((layer20, layer60), dim=1)
                layer62 = F.relu(self.up_bn11(nn.functional.conv2d(layer61, self.up_weights1, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer62, self.up_sample2, stride=2, bias=None)))
                layer71 = torch.cat((layer10, layer70), dim=1)
                layer72 = F.relu(self.up_bn21(nn.functional.conv2d(layer71, self.up_weights2, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer72, self.up_sample3, stride=2, bias=None)))
                layer81 = torch.cat((layer00, layer80), dim=1)
                layer82 = F.relu(self.up_bn31(nn.functional.conv2d(layer81, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn1(nn.functional.conv2d(layer82, self.gate_weights1, stride=1, padding=1, bias=None)))  # [1, 1, 50, 20]
                gate_prt = layer_out.view(self.out_features*self.in_features, 1)
                gate = layer_out.view(self.out_features, self.in_features, 1, 1)
                new_weights = self.weights.mul(gate)
                output = nn.functional.conv2d(input, new_weights, stride=1, bias=None)
                output = self.relu(output)
            elif epoch >= 4 and epoch % 2 != 0:
                self.weights.requires_grad = True
                self.gate_weights00.requires_grad = False
                self.gate_weights01.requires_grad = False
                self.gate_weights02.requires_grad = False
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.gate_weights1.requires_grad = False
                weight = self.weights.view(self.out_features, self.in_features, 5, 5)
                layer = F.relu(self.gate_bn00(nn.functional.conv2d(weight, self.gate_weights00, stride=1, bias=None)))
                layer = F.relu(self.gate_bn01(nn.functional.conv2d(layer, self.gate_weights01, stride=1, bias=None)))
                layer = F.relu(self.gate_bn02(nn.functional.conv2d(layer, self.gate_weights02, stride=1, bias=None)))
                data = layer.view(1, 1, self.out_features, self.in_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 3, 31]
                layer50 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer51 = torch.cat((layer30, layer50), dim=1)
                layer52 = F.relu(self.up_bn01(nn.functional.conv2d(layer51, self.up_weights0, stride=1, padding=1, bias=None)))
                layer60 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer52, self.up_sample1, stride=2, bias=None)))
                layer61 = torch.cat((layer20, layer60), dim=1)
                layer62 = F.relu(self.up_bn11(nn.functional.conv2d(layer61, self.up_weights1, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer62, self.up_sample2, stride=2, bias=None)))
                layer71 = torch.cat((layer10, layer70), dim=1)
                layer72 = F.relu(self.up_bn21(nn.functional.conv2d(layer71, self.up_weights2, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer72, self.up_sample3, stride=2, bias=None)))
                layer81 = torch.cat((layer00, layer80), dim=1)
                layer82 = F.relu(self.up_bn31(nn.functional.conv2d(layer81, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn1(nn.functional.conv2d(layer82, self.gate_weights1, stride=1, padding=1, bias=None)))  # [1, 1, 50, 20]
                gate_prt = layer_out.view(self.out_features*self.in_features, 1)
                gate = layer_out.view(self.out_features, self.in_features, 1, 1)
                new_weights = self.weights.mul(gate)
                output = nn.functional.conv2d(input, new_weights, stride=1, bias=None)
                output = self.relu(output)
        elif not self.training:
            if epoch < 4:
                gate_prt = torch.zeros(self.out_features * self.in_features, 1).cuda()
                output = nn.functional.conv2d(input, self.weights, stride=1, bias=None)
                output = self.relu(output)
            else:
                weight = self.weights.view(self.out_features, self.in_features, 5, 5)
                layer = F.relu(self.gate_bn00(nn.functional.conv2d(weight, self.gate_weights00, stride=1, bias=None)))
                layer = F.relu(self.gate_bn01(nn.functional.conv2d(layer, self.gate_weights01, stride=1, bias=None)))
                layer = F.relu(self.gate_bn02(nn.functional.conv2d(layer, self.gate_weights02, stride=1, bias=None)))
                data = layer.view(1, 1, self.out_features, self.in_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 3, 31]
                layer50 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer51 = torch.cat((layer30, layer50), dim=1)
                layer52 = F.relu(self.up_bn01(nn.functional.conv2d(layer51, self.up_weights0, stride=1, padding=1, bias=None)))
                layer60 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer52, self.up_sample1, stride=2, bias=None)))
                layer61 = torch.cat((layer20, layer60), dim=1)
                layer62 = F.relu(self.up_bn11(nn.functional.conv2d(layer61, self.up_weights1, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer62, self.up_sample2, stride=2, bias=None)))
                layer71 = torch.cat((layer10, layer70), dim=1)
                layer72 = F.relu(self.up_bn21(nn.functional.conv2d(layer71, self.up_weights2, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer72, self.up_sample3, stride=2, bias=None)))
                layer81 = torch.cat((layer00, layer80), dim=1)
                layer82 = F.relu(self.up_bn31(nn.functional.conv2d(layer81, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.gate_bn1(nn.functional.conv2d(layer82, self.gate_weights1, stride=1, padding=1, bias=None)))  # [1, 1, 50, 20]
                gate_prt = layer_out.view(self.out_features * self.in_features, 1)
                gate = layer_out.view(self.out_features, self.in_features, 1, 1)
                new_weights = self.weights.mul(gate)
                output = nn.functional.conv2d(input, new_weights, stride=1, bias=None)
                output = self.relu(output)
        return output, gate_prt


class Fully_Connect0(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect0, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.dw_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_weights3 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.dw_weights4 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.dw_weights5 = Parameter(torch.Tensor(256, 128, 3, 3))
        self.dw_bn5 = nn.BatchNorm2d(256)
        self.up_sample0 = Parameter(torch.Tensor(256, 128, 2, 3))
        self.up_bn00 = nn.BatchNorm2d(128)
        self.up_weights0 = Parameter(torch.Tensor(128, 256, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(128)
        self.up_sample1 = Parameter(torch.Tensor(128, 64, 2, 2))
        self.up_bn10 = nn.BatchNorm2d(64)
        self.up_weights1 = Parameter(torch.Tensor(64, 128, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(64)
        self.up_sample2 = Parameter(torch.Tensor(64, 32, 2, 3))
        self.up_bn20 = nn.BatchNorm2d(32)
        self.up_weights2 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(32)
        self.up_sample3 = Parameter(torch.Tensor(32, 16, 2, 2))
        self.up_bn30 = nn.BatchNorm2d(16)
        self.up_weights3 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn31 = nn.BatchNorm2d(16)
        self.up_sample4 = Parameter(torch.Tensor(16, 8, 2, 2))
        self.up_bn40 = nn.BatchNorm2d(8)
        self.up_weights4 = Parameter(torch.Tensor(8, 16, 3, 3))
        self.up_bn41 = nn.BatchNorm2d(8)
        self.weights0 = Parameter(torch.Tensor(1, 8, 3, 3))
        self.bn0 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weights,  0, 0.01)
        init.kaiming_normal_(self.dw_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights4, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights5, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample4, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights4, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weights0, mode='fan_out', nonlinearity='relu')
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input, epoch):
        if self.training:
            if epoch < 4:
                self.weights.requires_grad = True
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.dw_weights5.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_sample4.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.up_weights4.requires_grad = False
                self.weights0.requires_grad = False
                gate_prt = torch.zeros(self.in_features*self.out_features, 1).cuda()
                output = input.mm(self.weights)
            elif epoch >= 4 and epoch % 2 == 0:
                self.weights.requires_grad = False
                self.dw_weights0.requires_grad = True
                self.dw_weights1.requires_grad = True
                self.dw_weights2.requires_grad = True
                self.dw_weights3.requires_grad = True
                self.dw_weights4.requires_grad = True
                self.dw_weights5.requires_grad = True
                self.up_sample0.requires_grad = True
                self.up_sample1.requires_grad = True
                self.up_sample2.requires_grad = True
                self.up_sample3.requires_grad = True
                self.up_sample4.requires_grad = True
                self.up_weights0.requires_grad = True
                self.up_weights1.requires_grad = True
                self.up_weights2.requires_grad = True
                self.up_weights3.requires_grad = True
                self.up_weights4.requires_grad = True
                self.weights0.requires_grad = True
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))
                layer41 = self.maxpool(layer40)
                layer50 = F.relu(self.dw_bn5(nn.functional.conv2d(layer41, self.dw_weights5, stride=1, padding=1, bias=None)))  # [1, 256, 25, 15]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer50, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer40, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer30, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=2, bias=None)))
                layer81 = torch.cat((layer20, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=2, bias=None)))
                layer91 = torch.cat((layer10, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer100 = F.relu(self.up_bn40(nn.functional.conv_transpose2d(layer92, self.up_sample4, stride=2, bias=None)))
                layer101 = torch.cat((layer00, layer100), dim=1)
                layer102 = F.relu(self.up_bn41(nn.functional.conv2d(layer101, self.up_weights4, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.bn0(nn.functional.conv2d(layer102, self.weights0, stride=1, padding=1, bias=None)))  # [1, 1, 800, 500]
                gate_prt = layer_out.view(self.in_features*self.out_features, 1)
                gate = layer_out.view(self.in_features, self.out_features)
                new_weights = self.weights.mul(gate)
                output = input.mm(new_weights)
            elif epoch >= 4 and epoch % 2 != 0:
                self.weights.requires_grad = True
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.dw_weights5.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_sample4.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.up_weights4.requires_grad = False
                self.weights0.requires_grad = False
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))
                layer41 = self.maxpool(layer40)
                layer50 = F.relu(self.dw_bn5(nn.functional.conv2d(layer41, self.dw_weights5, stride=1, padding=1, bias=None)))  # [1, 256, 25, 15]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer50, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer40, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer30, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=2, bias=None)))
                layer81 = torch.cat((layer20, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=2, bias=None)))
                layer91 = torch.cat((layer10, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer100 = F.relu(self.up_bn40(nn.functional.conv_transpose2d(layer92, self.up_sample4, stride=2, bias=None)))
                layer101 = torch.cat((layer00, layer100), dim=1)
                layer102 = F.relu(self.up_bn41(nn.functional.conv2d(layer101, self.up_weights4, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.bn0(nn.functional.conv2d(layer102, self.weights0, stride=1, padding=1, bias=None)))  # [1, 1, 800, 500]
                gate_prt = layer_out.view(self.in_features*self.out_features, 1)
                gate = layer_out.view(self.in_features, self.out_features)
                new_weights = self.weights.mul(gate)
                output = input.mm(new_weights)
        elif not self.training:
            if epoch < 4:
                gate_prt = torch.zeros(self.in_features * self.out_features, 1).cuda()
                output = input.mm(self.weights)
            else:
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))
                layer41 = self.maxpool(layer40)
                layer50 = F.relu(self.dw_bn5(nn.functional.conv2d(layer41, self.dw_weights5, stride=1, padding=1, bias=None)))  # [1, 256, 25, 15]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer50, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer40, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer30, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=2, bias=None)))
                layer81 = torch.cat((layer20, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=2, bias=None)))
                layer91 = torch.cat((layer10, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer100 = F.relu(self.up_bn40(nn.functional.conv_transpose2d(layer92, self.up_sample4, stride=2, bias=None)))
                layer101 = torch.cat((layer00, layer100), dim=1)
                layer102 = F.relu(self.up_bn41(nn.functional.conv2d(layer101, self.up_weights4, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.bn0(nn.functional.conv2d(layer102, self.weights0, stride=1, padding=1, bias=None)))  # [1, 1, 800, 500]
                gate_prt = layer_out.view(self.in_features * self.out_features, 1)
                gate = layer_out.view(self.in_features, self.out_features)
                new_weights = self.weights.mul(gate)
                output = input.mm(new_weights)
        return output, gate_prt


class Fully_Connect1(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        """
        super(Fully_Connect1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.dw_weights0 = Parameter(torch.Tensor(8, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(8)
        self.dw_weights1 = Parameter(torch.Tensor(16, 8, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_weights2 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_weights3 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.dw_weights4 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.up_sample0 = Parameter(torch.Tensor(128, 64, 2, 2))
        self.up_bn00 = nn.BatchNorm2d(64)
        self.up_weights0 = Parameter(torch.Tensor(64, 128, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(64)
        self.up_sample1 = Parameter(torch.Tensor(64, 32, 3, 2))
        self.up_bn10 = nn.BatchNorm2d(32)
        self.up_weights1 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(32)
        self.up_sample2 = Parameter(torch.Tensor(32, 16, 2, 2))
        self.up_bn20 = nn.BatchNorm2d(16)
        self.up_weights2 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(16)
        self.up_sample3 = Parameter(torch.Tensor(16, 8, 2, 2))
        self.up_bn30 = nn.BatchNorm2d(8)
        self.up_weights3 = Parameter(torch.Tensor(8, 16, 3, 3))
        self.up_bn31 = nn.BatchNorm2d(8)
        self.weights0 = Parameter(torch.Tensor(1, 8, 3, 3))
        self.bn0 = nn.BatchNorm2d(1)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weights,  0, 0.01)
        init.kaiming_normal_(self.dw_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights4, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weights0, mode='fan_out', nonlinearity='relu')
        if self.use_bias:
            self.bias.data.fill_(0)

    def forward(self, input, epoch):
        if self.training:
            if epoch < 4:
                self.weights.requires_grad = True
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.weights0.requires_grad = False
                gate_prt = torch.zeros(self.in_features * self.out_features, 1).cuda()
                output = input.mm(self.weights)
            elif epoch >= 4 and epoch % 2 == 0:
                self.weights.requires_grad = False
                self.dw_weights0.requires_grad = True
                self.dw_weights1.requires_grad = True
                self.dw_weights2.requires_grad = True
                self.dw_weights3.requires_grad = True
                self.dw_weights4.requires_grad = True
                self.up_sample0.requires_grad = True
                self.up_sample1.requires_grad = True
                self.up_sample2.requires_grad = True
                self.up_sample3.requires_grad = True
                self.up_weights0.requires_grad = True
                self.up_weights1.requires_grad = True
                self.up_weights2.requires_grad = True
                self.up_weights3.requires_grad = True
                self.weights0.requires_grad = True
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool0(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool0(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool1(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool1(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 31, 2]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer30, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer20, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=(2, 1), bias=None)))
                layer81 = torch.cat((layer10, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=(2, 1), bias=None)))
                layer91 = torch.cat((layer00, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.bn0(nn.functional.conv2d(layer92, self.weights0, stride=1, padding=1, bias=None)))  # [1, 1, 500, 10]
                gate_prt = layer_out.view(self.in_features * self.out_features, 1)
                gate = layer_out.view(self.in_features, self.out_features)
                new_weights = self.weights.mul(gate)
                output = input.mm(new_weights)
            elif epoch >= 4 and epoch % 2 != 0:
                self.weights.requires_grad = True
                self.dw_weights0.requires_grad = False
                self.dw_weights1.requires_grad = False
                self.dw_weights2.requires_grad = False
                self.dw_weights3.requires_grad = False
                self.dw_weights4.requires_grad = False
                self.up_sample0.requires_grad = False
                self.up_sample1.requires_grad = False
                self.up_sample2.requires_grad = False
                self.up_sample3.requires_grad = False
                self.up_weights0.requires_grad = False
                self.up_weights1.requires_grad = False
                self.up_weights2.requires_grad = False
                self.up_weights3.requires_grad = False
                self.weights0.requires_grad = False
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool0(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool0(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool1(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool1(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 31, 2]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer30, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer20, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=(2, 1), bias=None)))
                layer81 = torch.cat((layer10, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=(2, 1), bias=None)))
                layer91 = torch.cat((layer00, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.bn0(nn.functional.conv2d(layer92, self.weights0, stride=1, padding=1, bias=None)))  # [1, 1, 500, 10]
                gate_prt = layer_out.view(self.in_features * self.out_features, 1)
                gate = layer_out.view(self.in_features, self.out_features)
                new_weights = self.weights.mul(gate)
                output = input.mm(new_weights)
        elif not self.training:
            if epoch < 4:
                gate_prt = torch.zeros(self.in_features * self.out_features, 1).cuda()
                output = input.mm(self.weights)
            else:
                data = self.weights.view(1, 1, self.in_features, self.out_features)
                layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(data, self.dw_weights0, stride=1, padding=1, bias=None)))
                layer01 = self.maxpool0(layer00)
                layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
                layer11 = self.maxpool0(layer10)
                layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
                layer21 = self.maxpool1(layer20)
                layer30 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))
                layer31 = self.maxpool1(layer30)
                layer40 = F.relu(self.dw_bn4(nn.functional.conv2d(layer31, self.dw_weights4, stride=1, padding=1, bias=None)))  # [1, 128, 31, 2]
                layer60 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer40, self.up_sample0, stride=2, bias=None)))
                layer61 = torch.cat((layer30, layer60), dim=1)
                layer62 = F.relu(self.up_bn01(nn.functional.conv2d(layer61, self.up_weights0, stride=1, padding=1, bias=None)))
                layer70 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer62, self.up_sample1, stride=2, bias=None)))
                layer71 = torch.cat((layer20, layer70), dim=1)
                layer72 = F.relu(self.up_bn11(nn.functional.conv2d(layer71, self.up_weights1, stride=1, padding=1, bias=None)))
                layer80 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer72, self.up_sample2, stride=(2, 1), bias=None)))
                layer81 = torch.cat((layer10, layer80), dim=1)
                layer82 = F.relu(self.up_bn21(nn.functional.conv2d(layer81, self.up_weights2, stride=1, padding=1, bias=None)))
                layer90 = F.relu(self.up_bn30(nn.functional.conv_transpose2d(layer82, self.up_sample3, stride=(2, 1), bias=None)))
                layer91 = torch.cat((layer00, layer90), dim=1)
                layer92 = F.relu(self.up_bn31(nn.functional.conv2d(layer91, self.up_weights3, stride=1, padding=1, bias=None)))
                layer_out = F.relu(self.bn0(nn.functional.conv2d(layer92, self.weights0, stride=1, padding=1, bias=None)))  # [1, 1, 500, 10]
                gate_prt = layer_out.view(self.in_features * self.out_features, 1)
                gate = layer_out.view(self.in_features, self.out_features)
                new_weights = self.weights.mul(gate)
                output = input.mm(new_weights)
        return output, gate_prt