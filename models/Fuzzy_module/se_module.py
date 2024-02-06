from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, min_channels=1): # maybe you need to modify min_channels
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, min_channels), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, min_channels), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_3d(nn.Module):
    def __init__(self, channel, reduction=4, min_channels=1): # maybe you need to modify min_channels
        super(SELayer_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, min_channels), bias=False),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(max(channel // reduction, min_channels), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, dummy_tensor=None):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SELayer_dual_3d(nn.Module):
    def __init__(self, channel, reduction=4, min_channels=1, expand=1, max_channels=1024): # maybe you need to modify min_channels
        super(SELayer_dual_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.expand = expand
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, min_channels), bias=False),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(max(channel // reduction, min_channels), channel, bias=False),
            nn.Sigmoid()
        )

        if expand > 1:
            self.fc_ex = nn.Sequential(
                nn.Linear(channel, min(channel * expand, max_channels), bias=False),
                # nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Linear(min(channel * expand, max_channels), channel, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        if self.expand > 1:
            y = self.fc(y).view(b, c, 1, 1, 1) + self.fc_ex(y).view(b, c, 1, 1, 1)
        else:
            y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
