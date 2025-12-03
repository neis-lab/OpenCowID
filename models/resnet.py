import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Orig_resnet_with_projection(nn.Module):
    def __init__(self):
        super(Orig_resnet_with_projection, self).__init__()
        # for older pytorch
        self.encoder = models.resnet50(pretrained=True)

        self.projection_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(1000, 128),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        x = F.normalize(x, dim=1)
        return x
