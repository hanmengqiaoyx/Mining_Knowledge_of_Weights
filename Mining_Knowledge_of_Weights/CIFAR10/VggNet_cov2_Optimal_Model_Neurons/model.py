from torch import nn
from layer import Fully_Connect0, Fully_Connect1, Fully_Connect2


class VggNet_cov2(nn.Module):
    def __init__(self, fc_in=64*16*16, fc_dims1=256, fc_dims2=256, num_classes=10):
        super(VggNet_cov2, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16
        )
        self.fc_in = fc_in
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.layer0 = Fully_Connect0(self.fc_in, self.fc_dims1)
        self.layer1 = Fully_Connect1(self.fc_dims1, self.fc_dims2)
        self.layer2 = Fully_Connect2(self.fc_dims2, num_classes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, epoch):
        o = self.features(x)
        o = o.view(o.size(0), -1)
        # 3
        a, gate_prt0 = self.layer0(o, epoch)
        a_out = self.relu(a)
        # 4
        b, gate_prt1 = self.layer1(a_out, epoch)
        b_out = self.relu(b)
        # 5
        c, gate_prt2 = self.layer2(b_out, epoch)
        return c, gate_prt0, gate_prt1, gate_prt2