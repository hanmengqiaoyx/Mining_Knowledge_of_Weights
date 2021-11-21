from torch import nn
from layer import Fully_Connect0, Fully_Connect1, Fully_Connect2


class MLP(nn.Module):
    def __init__(self, fc_in=784, fc_dims1=300, fc_dims2=100, class_num=10):
        super(MLP, self).__init__()
        self.fc_in = fc_in
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.layer0 = Fully_Connect0(self.fc_in, self.fc_dims1)
        self.layer1 = Fully_Connect1(self.fc_dims1, self.fc_dims2)
        self.layer2 = Fully_Connect2(self.fc_dims2, class_num)
        self.relu = nn.ReLU()

    def forward(self, x, epoch):
        o = x.view(x.size(0), -1)
        # 1
        out, f_gate0 = self.layer0(o, epoch)
        out = self.relu(out)
        # 2
        out, f_gate1 = self.layer1(out, epoch)
        out = self.relu(out)
        # 3
        out, f_gate2 = self.layer2(out, epoch)
        return out, f_gate0, f_gate1, f_gate2