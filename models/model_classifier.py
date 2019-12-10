import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = 0.5

        def convbn(ci, co, ksz, s=1, pz=0):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        self.m = nn.Sequential(
            # nn.Dropout(0.2),
            convbn(3, 64, 3, 1, 1),
            convbn(64, 64, 3, 1, 1),
            convbn(64, 64, 3, 1, 1),
            convbn(64, 64, 3, 1, 1),
            nn.Dropout(d),
            nn.AvgPool2d(2),
            convbn(64, 128, 3, 1, 1),
            convbn(128, 128, 3, 1, 1),
            convbn(128, 128, 3, 1, 1),
            nn.Dropout(d),
            convbn(128, 256, 3, 1, 1),
            convbn(256, 256, 3, 1, 1),
            convbn(256, 256, 1, 2, 1),
            nn.AvgPool2d(8),
            View(6144),
            nn.Linear(6144, 4),
            nn.Sigmoid())

    def forward(self, x):
        return self.m(x)
