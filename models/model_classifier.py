import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


class ConvModel(nn.Module):
    def __init__(self, dropval=0.5, l1=32, l2=64, l3=128):
        super().__init__()
        d = dropval  # Prob tat element is zeroes

        def convbn(ci, co, ksz, s=1, pz=0):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        self.m = nn.Sequential(
            convbn(3, l1, 3, 1, 1),
            convbn(l1, l1, 3, 1, 1),
            convbn(l1, l1, 3, 1, 1),
            convbn(l1, l1, 3, 1, 1),
            nn.Dropout(d),
            nn.AvgPool2d(2),
            convbn(l1, l2, 3, 1, 1),
            convbn(l2, l2, 3, 1, 1),
            convbn(l2, l2, 3, 1, 1),
            nn.Dropout(d),
            convbn(l2, l3, 3, 1, 1),
            convbn(l3, l3, 3, 1, 1),
            convbn(l3, l3, 1, 2, 1),
            nn.AvgPool2d(8),
            View(6144),
            nn.Linear(6144, 4),
            nn.Sigmoid())

    def forward(self, x):
        return self.m(x)
