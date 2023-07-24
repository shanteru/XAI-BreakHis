'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

from email.policy import strict
import torchvision.models as models
import torch.nn as nn
from torchvision.models.resnet import Bottleneck,_resnet
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet50_SSL(nn.Module):
    def __init__(self, projector, supervised_pretrained=None):
        super(ResNet50_SSL, self).__init__()
        print(projector, supervised_pretrained)

        # Backbone: ResNet50
        self.backbone = models.resnet50(pretrained = supervised_pretrained)
        self.backbone.fc = nn.Identity()

        # Projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # Normalization layer for the representations
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.backbone(x))
        return F.normalize(self.bn(z), dim = -1)
